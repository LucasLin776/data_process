"""螺栓声音数据增强：真实背景混合、三频段随机均衡和来源审计。

所有合成数据仅用于训练；不自动生成验证集或测试集。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.signal import firwin2, oaconvolve, resample_poly, stft

VERSION = 1
AUDITION_COLUMNS = ["Name", "Start", "Duration", "Time Format", "Type", "Description"]
PLOT_SPEC = {
    "waveform_directory": "waveforms", "spectrogram_directory": "spectrograms",
    "stft_window": "hann", "stft_window_samples": 1024, "stft_hop_samples": 256,
    "spectrogram_db_range": [-100, 0], "spectrogram_reference": "STFT amplitude relative to 1 FS",
    "waveform_envelope_bins": 3000, "dpi": 120,
}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signal_quality(x: np.ndarray) -> dict:
    return {
        "rms": float(np.sqrt(np.mean(np.square(x)))),
        "peak": float(np.max(np.abs(x))),
        "dc_mean": float(np.mean(x)),
        "near_full_scale_fraction": float(np.mean(np.abs(x) >= 0.999)),
    }


def read_audio(path: Path, target_rate: int, channel: int = 0) -> tuple[np.ndarray, dict]:
    """选择单个声道，不混合声道；去直流，抗混叠重采样，不做峰值归一化。"""
    info = sf.info(path)
    raw, native_rate = sf.read(path, dtype="float64", always_2d=True)
    if not len(raw) or not 0 <= channel < raw.shape[1]:
        raise ValueError(f"空音频或声道不存在: {path}")
    x = raw[:, channel]
    if not np.isfinite(x).all():
        raise ValueError(f"音频含 NaN/Inf: {path}")
    quality = {
        "path": str(path), "sha256": sha256(path), "native_sample_rate": native_rate,
        "native_frames": len(x), "duration_s": len(x) / native_rate,
        "channels": info.channels, "subtype": info.subtype, "selected_channel": channel,
        **signal_quality(x),
    }
    x = x - quality["dc_mean"]
    if native_rate != target_rate:
        divisor = math.gcd(native_rate, target_rate)
        x = resample_poly(x, target_rate // divisor, native_rate // divisor)
    quality["processing_sample_rate"] = target_rate
    quality["warnings"] = []
    if quality["near_full_scale_fraction"] > 0:
        quality["warnings"].append("源文件存在近满量程样本，可能削波；降低音量不能修复源失真")
    if np.max(np.abs(x)) > 1:
        quality["warnings"].append("去直流/重采样后峰值超过1，写出前将统一衰减，不能硬裁剪")
    return np.ascontiguousarray(x), quality


def decimal_seconds(value: str) -> Decimal:
    value = value.strip()
    if not re.fullmatch(r"\d+(?::\d{1,2}){0,2}(?:\.\d+)?", value):
        raise ValueError(f"不支持的时间: {value!r}，使用秒、分:秒或时:分:秒")
    parts = [Decimal(part) for part in value.split(":")]
    if any(part >= 60 for part in parts[1:]):
        raise ValueError(f"时间的秒/非首位分钟必须小于60: {value}")
    seconds = Decimal(0)
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


def validate_events(events: list[dict], duration: float) -> list[dict]:
    if not isinstance(events, list) or not events:
        raise ValueError("正样本必须有非空事件区间，不能根据波形自动猜标签")
    result = []
    for row in events:
        start, end = float(row["start_s"]), float(row["end_s"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
            raise ValueError(f"事件越界或时长为0: {row}; 音频时长={duration}")
        result.append({"start_s": start, "end_s": end})
    result.sort(key=lambda row: row["start_s"])
    if any(right["start_s"] < left["end_s"] for left, right in pairwise(result)):
        raise ValueError("事件区间重叠，请先合并同一事件的标注")
    return result


def read_markers(path: Path, duration: float) -> list[dict]:
    """兼容用户的 Audition UTF-8 BOM + Tab CSV，也接受逗号分隔。"""
    text = path.read_text(encoding="utf-8-sig").lstrip("\r\n")
    first = next((line for line in text.splitlines() if line.strip()), "")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t" if "\t" in first else ",")
    names = [name.strip() for name in (reader.fieldnames or [])]
    if len(set(names)) != len(names) or not {"Name", "Start", "Duration", "Time Format"} <= set(names):
        raise ValueError(f"Audition表头不正确: {path}")
    reader.fieldnames = names
    events = []
    for row in reader:
        if None in row:
            raise ValueError(f"标记列数不正确: {path}:{reader.line_num}")
        if not any(value and value.strip() for value in row.values()):
            continue
        if row["Name"].strip() != "bolt_drop" or row["Time Format"].strip().lower() != "decimal":
            raise ValueError("标记名称必须为bolt_drop，Time Format必须为decimal")
        start = decimal_seconds(row["Start"])
        span = decimal_seconds(row["Duration"])
        if span <= 0:
            raise ValueError("Duration必须大于0，请导出范围标记")
        end = start + span
        if Decimal(str(duration)) < end <= Decimal(str(duration)) + Decimal("0.001"):
            end = Decimal(str(duration))
        events.append({"start_s": float(start), "end_s": float(end)})
    return validate_events(events, duration)


def export_markers(path: Path, events: list[dict]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\r\n")
        writer.writerow(AUDITION_COLUMNS)
        for event in events:
            start, end = Decimal(str(event["start_s"])), Decimal(str(event["end_s"]))
            writer.writerow(["bolt_drop", format(start, "f"), format(end - start, "f"),
                             "decimal", "Cue", "Derived from confirmed source annotation"])


@dataclass
class Source:
    source_id: str
    role: str
    path: Path
    x: np.ndarray
    quality: dict
    events: list[dict]
    annotation: dict | None


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve(strict=True)


def random_eq_settings(cfg: dict) -> dict:
    """旧配置缺少random_eq时保持关闭；开启时严格校验频率与增益。"""
    raw = cfg.get("random_eq", {})
    if not isinstance(raw, dict):
        raise ValueError("random_eq必须是对象")
    settings = {
        "enabled": False, "gain_db": [-3., 3.], "crossovers_hz": [500., 4000.],
        "transition_fraction": .2, "fir_taps": 513, "source_variants": 1,
        **raw,
    }
    if type(settings["enabled"]) is not bool:
        raise ValueError("random_eq.enabled必须是布尔值")
    if not settings["enabled"]:
        return settings

    def finite_number(value: object) -> bool:
        return type(value) in (int, float) and math.isfinite(value)

    gains = settings["gain_db"]
    if (not isinstance(gains, list) or len(gains) != 2
            or not all(finite_number(value) for value in gains)
            or not -6 <= gains[0] <= gains[1] <= 6):
        raise ValueError("random_eq.gain_db必须是[-6,6]dB范围内的[最小值,最大值]")
    fraction = settings["transition_fraction"]
    if not finite_number(fraction) or not 0 < fraction < .5:
        raise ValueError("random_eq.transition_fraction必须介于0和0.5之间")
    crossovers = settings["crossovers_hz"]
    if (not isinstance(crossovers, list) or len(crossovers) != 2
            or not all(finite_number(value) for value in crossovers)):
        raise ValueError("random_eq.crossovers_hz必须是两个有限频率")
    low, high = crossovers
    if not (0 < low * (1 - fraction) < low * (1 + fraction)
            < high * (1 - fraction) < high * (1 + fraction) < cfg["sample_rate"] / 2):
        raise ValueError("random_eq频段过渡带必须递增、不重叠且低于奈奎斯特频率")
    taps = settings["fir_taps"]
    if type(taps) is not int or not 33 <= taps <= 8193 or taps % 2 == 0:
        raise ValueError("random_eq.fir_taps必须是33～8193之间的奇数")
    variants = settings["source_variants"]
    if type(variants) is not int or not 1 <= variants <= 100:
        raise ValueError("random_eq.source_variants必须是1～100之间的整数")
    return settings


def sample_equalizer(cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray | None, dict]:
    """三段独立抽样；用SciPy线性相位FIR和连续过渡带避免频域硬切。"""
    settings = random_eq_settings(cfg)
    if not settings["enabled"]:
        return None, {"applied": False}
    sr = cfg["sample_rate"]
    low, high = settings["crossovers_hz"]
    fraction = settings["transition_fraction"]
    gains_db = rng.uniform(*settings["gain_db"], size=3)
    gains = 10 ** (gains_db / 20)
    frequencies = [0., low * (1 - fraction), low * (1 + fraction),
                   high * (1 - fraction), high * (1 + fraction), sr / 2]
    taps = firwin2(settings["fir_taps"], frequencies, np.repeat(gains, 2), fs=sr, window="hann")
    half = (len(taps) - 1) // 2
    return taps, {
        "applied": True, "method": "scipy_firwin2_hann_linear_phase",
        "gain_db": dict(zip(("low", "mid", "high"), map(float, gains_db))),
        "crossovers_hz": list(settings["crossovers_hz"]),
        "transition_edges_hz": [frequencies[1:3], frequencies[3:5]],
        "fir_taps": len(taps), "delay_compensated_samples": half,
        "maximum_temporal_spread_s": half / sr, "boundary_padding": "reflect",
        "sample_rate": sr, "event_time_mapping": "unchanged; filter delay compensated",
    }


def apply_equalizer(x: np.ndarray, taps: np.ndarray | None) -> np.ndarray:
    """对称填充后valid卷积消除FIR延迟，保持长度；不做归一化或裁剪。"""
    if taps is None:
        return x
    half = (len(taps) - 1) // 2
    padded = np.pad(x, (half, half), mode="reflect")
    return np.ascontiguousarray(oaconvolve(padded, taps, mode="valid"))


def load_config(path: Path) -> tuple[dict, list[Source]]:
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if cfg.get("version") != VERSION:
        raise ValueError("不支持的配置版本")
    sr = cfg["sample_rate"]
    if type(sr) is not int or not 1000 <= sr <= 192000:
        raise ValueError("sample_rate必须为1000～192000的整数")
    random_eq_settings(cfg)
    duration = cfg["duration_s"]
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_s必须为正数")
    if type(cfg["channel"]) is not int or cfg["channel"] < 0:
        raise ValueError("channel必须为非负整数")
    if type(cfg["seed"]) is not int or cfg["seed"] < 0:
        raise ValueError("seed必须为非负整数")
    if type(cfg["repeats_per_snr"]) is not int or cfg["repeats_per_snr"] <= 0:
        raise ValueError("repeats_per_snr必须为正整数")
    if not isinstance(cfg["snr_db"], list) or not cfg["snr_db"] or not all(
        isinstance(value, (int, float)) and math.isfinite(value) and -40 <= value <= 40
        for value in cfg["snr_db"]
    ):
        raise ValueError("snr_db必须是-40～40之间的非空数值列表")
    gain_range = cfg["output_gain_db"]
    if (not isinstance(gain_range, list) or len(gain_range) != 2
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in gain_range)
            or not -60 <= gain_range[0] <= gain_range[1] <= 0):
        raise ValueError("output_gain_db必须为[-60,0]范围内的[最小值,最大值]")
    if not isinstance(cfg["peak_limit"], (int, float)) or not 0 < cfg["peak_limit"] < .999:
        raise ValueError("peak_limit必须介于0和0.999之间")
    for key in ("context_s", "fade_s"):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"{key}必须为非负有限数")
    group_id = cfg["group_id"]
    if not isinstance(group_id, str) or not group_id.strip():
        raise ValueError("必须填写group_id；当前来源及其增强版本全部归入同一训练组")
    if not cfg.get("foregrounds") or not cfg.get("backgrounds"):
        raise ValueError("至少需要一条已标注正样本和一条确认正常背景")
    sources, ids, hashes = [], set(), {}
    for key, role in (("foregrounds", "foreground"), ("backgrounds", "background"),
                      ("hard_negatives", "hard_negative")):
        for row in cfg.get(key, []):
            source_id = row["id"]
            if not isinstance(source_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", source_id):
                raise ValueError("来源id只能使用英文字母、数字、下划线和短横线")
            if source_id in ids:
                raise ValueError(f"来源id重复: {source_id}")
            ids.add(source_id)
            if row.get("reviewed") is not True or row.get("split", "train") != "train":
                raise ValueError(f"{source_id}: 必须已人工确认，且仅可使用train来源")
            audio_path = resolve_path(row["audio"], path.parent)
            x, quality = read_audio(audio_path, sr, cfg["channel"])
            if quality["sha256"] in hashes:
                raise ValueError(f"重复源音频: {source_id}与{hashes[quality['sha256']]}")
            hashes[quality["sha256"]] = source_id
            events, annotation = [], None
            if role == "foreground":
                if ("markers" in row) == ("events" in row):
                    raise ValueError("正样本必须且只能提供markers或events其中之一")
                if "markers" in row:
                    marker_path = resolve_path(row["markers"], path.parent)
                    events = read_markers(marker_path, quality["duration_s"])
                    annotation = {"path": str(marker_path), "sha256": sha256(marker_path),
                                  "format": "audition"}
                else:
                    events = validate_events(row["events"], quality["duration_s"])
                    annotation = {"format": "inline_events"}
                # 保留原始正例时不能静默裁掉未混入的其他事件。
                if events[-1]["end_s"] > min(len(x) / sr, duration):
                    raise ValueError(f"{source_id}: 事件超出输出长度，请增大duration_s或单独准备录音")
            elif "markers" in row or row.get("events"):
                raise ValueError(f"{source_id}: 背景/干扰负例不能含螺栓事件标签")
            if len(x) < round(duration * sr):
                raise ValueError(f"{source_id}: 源音频短于{duration}秒，不自动循环或补零")
            sources.append(Source(source_id, role, audio_path, x, quality, events, annotation))
    return cfg, sources


def event_clip(source: Source, index: int, cfg: dict) -> tuple[np.ndarray, int, int]:
    """淡入淡出仅作用于标注之外的上下文，不修改事件冲击起点。"""
    sr = cfg["sample_rate"]
    event = source.events[index]
    left, right = round(event["start_s"] * sr), round(event["end_s"] * sr)
    if left >= right:
        raise ValueError("重采样后事件不足一个采样点")
    context = round(cfg["context_s"] * sr)
    previous_end = round(source.events[index - 1]["end_s"] * sr) if index else 0
    next_start = (round(source.events[index + 1]["start_s"] * sr)
                  if index + 1 < len(source.events) else len(source.x))
    first, stop = max(previous_end, left - context), min(next_start, right + context)
    clip = source.x[first:stop].copy()
    before, after = left - first, stop - right
    fade = round(cfg["fade_s"] * sr)
    n = min(fade, before)
    if n:
        clip[:n] *= np.linspace(0, 1, n)
    n = min(fade, after)
    if n:
        clip[-n:] *= np.linspace(1, 0, n)
    return clip, before, before + right - left


def mix_at_snr(background: np.ndarray, foreground: np.ndarray, start: int,
               active_start: int, active_end: int, snr_db: float,
               peak_limit: float, gain_db: float, *,
               eq_taps: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """背景/完整前景共享EQ，在EQ后设SNR；正负例共享输出增益。"""
    if not 0 <= start <= len(background) - len(foreground):
        raise ValueError("事件及其上下文放不进背景")
    if not 0 <= active_start < active_end <= len(foreground):
        raise ValueError("无效事件活动区间")
    active = slice(start + active_start, start + active_end)
    event = np.zeros_like(background)
    event[start:start + len(foreground)] = foreground
    # 先放进完整时间轴再滤波，不在裁出的事件边界截断滤波响应。
    background = apply_equalizer(background, eq_taps)
    event = apply_equalizer(event, eq_taps)
    signal_power = float(np.mean(np.square(event[active])))
    noise_power = float(np.mean(np.square(background[active])))
    if min(signal_power, noise_power) <= 1e-16:
        raise ValueError("事件或对应背景接近静音，不能按指定信噪比混合")
    multiplier = math.sqrt(noise_power / signal_power) * 10 ** (snr_db / 20)
    event *= multiplier
    mixed = background + event
    # 同时考虑正例和配对负例的峰值，保留相同背景音量，避免响度泄露标签。
    peak = max(float(np.max(np.abs(mixed))), float(np.max(np.abs(background))))
    headroom = min(1., peak_limit / peak) if peak else 1.
    gain = headroom * 10 ** (gain_db / 20)
    actual = 10 * math.log10(float(np.mean(np.square(event[active]))) / noise_power)
    return mixed * gain, background * gain, {
        "snr_db_requested": float(snr_db), "snr_db_achieved_pre_quantization": actual,
        "snr_definition": "foreground/background power over annotated event interval; source includes residual noise",
        "foreground_multiplier": multiplier, "output_gain_db_requested": float(gain_db),
        "headroom_gain": headroom, "common_output_gain": gain,
        "pre_mix_signal_power": signal_power, "pre_mix_background_power": noise_power,
        "snr_measured_after_eq": eq_taps is not None,
    }


def save_pcm(path: Path, x: np.ndarray, sr: int) -> dict:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖: {path}")
    if not np.isfinite(x).all() or np.max(np.abs(x)) >= .999:
        raise ValueError("输出含非有限值或接近量程边界，拒绝写出")
    sf.write(path, x, sr, subtype="PCM_16")
    written, _ = sf.read(path, dtype="float64")
    return signal_quality(written)


def render_audio_plots(audio: Path, output: Path, stem: str) -> dict:
    """直接绘制实际写出的WAV；包络保留峰值，STFT使用固定幅值参考和色标。"""
    x, sr = sf.read(audio, dtype="float64")
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError(f"绘图需要至少两个采样点的有限单声道音频: {audio}")
    paths = {"waveform": f"waveforms/{stem}.png", "spectrogram": f"spectrograms/{stem}.png"}
    for path in paths.values():
        target = output / path
        if target.exists():
            raise FileExistsError(f"拒绝覆盖图片: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
    duration = len(x) / sr
    fig = Figure(figsize=(12, 3.2), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    if len(x) <= PLOT_SPEC["waveform_envelope_bins"] * 2:
        ax.plot(np.arange(len(x)) / sr, x, linewidth=.6, color="#276b9b")
    else:
        # 逐块最小/最大值包络，避免简单抽点遗漏短时冲击。
        step = math.ceil(len(x) / PLOT_SPEC["waveform_envelope_bins"])
        starts = np.arange(0, len(x), step)
        ends = np.minimum(starts + step, len(x))
        centers = (starts + ends - 1) / (2 * sr)
        ax.fill_between(centers, np.minimum.reduceat(x, starts), np.maximum.reduceat(x, starts),
                        color="#276b9b", linewidth=0)
    ax.set(xlim=(0, duration), ylim=(-1.02, 1.02), xlabel="Time (s)",
           ylabel="Amplitude (FS)", title=f"{stem} - Waveform")
    ax.grid(alpha=.2)
    fig.savefig(output / paths["waveform"], dpi=PLOT_SPEC["dpi"])
    fig.clear()

    window = min(PLOT_SPEC["stft_window_samples"], len(x))
    hop = min(PLOT_SPEC["stft_hop_samples"], window)
    frequencies, times, spectrum = stft(
        x, fs=sr, window="hann", nperseg=window, noverlap=window - hop,
        boundary="zeros", padded=True,
    )
    db = 20 * np.log10(np.maximum(np.abs(spectrum), 1e-12))
    # 用STFT帧中心推算像素边界，避免把分析帧错误拉伸到整个音频时长。
    frequency_step = sr / window
    extent = [times[0] - hop / (2 * sr), times[-1] + hop / (2 * sr),
              (frequencies[0] - frequency_step / 2) / 1000,
              (frequencies[-1] + frequency_step / 2) / 1000]
    fig = Figure(figsize=(12, 4), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    picture = ax.imshow(db, origin="lower", aspect="auto", extent=extent, interpolation="nearest",
                        cmap="magma", vmin=PLOT_SPEC["spectrogram_db_range"][0],
                        vmax=PLOT_SPEC["spectrogram_db_range"][1])
    ax.set(xlim=(0, duration), ylim=(0, sr / 2000), xlabel="Time (s)",
           ylabel="Frequency (kHz)", title=f"{stem} - Spectrogram")
    fig.colorbar(picture, ax=ax, label="STFT magnitude (dB re 1 FS)")
    fig.savefig(output / paths["spectrogram"], dpi=PLOT_SPEC["dpi"])
    fig.clear()
    return paths


def inventory(input_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"拒绝覆盖报告: {output}")
    rows = []
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".wav":
            _, quality = read_audio(path.resolve(), sf.info(path).samplerate)
            quality["label"] = "unassigned; filenames do not establish labels"
            rows.append(quality)
    if not rows:
        raise ValueError("目录中没有WAV文件")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"files": len(rows), "recordings": rows}
    write_json(output, result)
    return {"files": len(rows), "report": str(output.resolve())}


def generate(config_path: Path, output: Path) -> dict:
    started = time.perf_counter()
    config_path = config_path.resolve(strict=True)
    cfg, sources = load_config(config_path)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录必须不存在，请换一个新目录: {output}")
    sr, frames = cfg["sample_rate"], round(cfg["duration_s"] * cfg["sample_rate"])
    foregrounds = [source for source in sources if source.role == "foreground"]
    backgrounds = [source for source in sources if source.role == "background"]
    # 所有源和事件先校验；没有标签、静音源或过长片段不能留下看似可用的数据集。
    clips = []
    for source in foregrounds:
        for index in range(len(source.events)):
            clip, left, right = event_clip(source, index, cfg)
            if len(clip) > frames or np.mean(np.square(clip[left:right])) <= 1e-16:
                raise ValueError(f"{source.source_id}: 事件过长或接近静音")
            clips.append((source, index, clip, left, right))
    for source in backgrounds:
        if np.mean(np.square(source.x[:frames])) <= 1e-16:
            raise ValueError(f"{source.source_id}: 正常背景接近静音，无法按SNR混合")
    output.mkdir(parents=True, exist_ok=False)
    (output / "audio").mkdir()
    (output / "labels").mkdir()
    write_json(output / "RUNNING.json", {"status": "incomplete", "notice": "Only summary.json status=complete indicates success"})
    write_json(output / "config_used.json", cfg)
    write_json(output / "sources.json", {
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "sources": [{"id": source.source_id, "role": source.role, **source.quality,
                     "events": source.events, "annotation": source.annotation} for source in sources],
    })
    rng = np.random.default_rng(cfg["seed"])
    # EQ使用独立随机流，开关EQ不会改变原有位置、背景裁剪和全局增益抽样。
    eq_rng = np.random.default_rng(np.random.SeedSequence([cfg["seed"], 0x4551]))
    eq_settings = random_eq_settings(cfg)
    records, logs = [], []

    def save(derivation_id: str, x: np.ndarray, events: list[dict], used: list[Source],
             details: dict) -> None:
        details = {"random_eq": {"applied": False}, **details}
        recording_id = f"{len(records) + 1:02d}"
        relative_audio = f"audio/{recording_id}.wav"
        quality = save_pcm(output / relative_audio, x, sr)
        plots = render_audio_plots(output / relative_audio, output, recording_id)
        labels = None
        if events:
            labels = f"labels/{recording_id}.csv"
            export_markers(output / labels, events)
        row = {
            "recording_id": recording_id, "audio": relative_audio,
            "derivation_id": derivation_id, "plots": plots,
            "group_id": cfg["group_id"], "split": "train", "reviewed": True,
            "events": events, "source_ids": [source.source_id for source in used],
            "source_sha256": [source.quality["sha256"] for source in used],
            "synthetic": details["kind"] == "positive_mixture", "labels": labels,
            "annotation": {"tool": "deterministic_transform_of_reviewed_sources",
                           "output_listened_by_human": False},
            "notes": "Training only. reviewed refers to confirmed source labels; transformed files are not individually listened. Independent real validation/test data required.",
        }
        records.append(row)
        logs.append({**row, "duration_s": len(x) / sr, "sample_rate": sr,
                     "augmentation": details, "output_quality": quality})

    # 保留一份统一格式的真实来源，不扩大真实试验数量。
    for source in sources:
        x = source.x[:frames].copy()
        peak = float(np.max(np.abs(x)))
        gain = min(1., cfg["peak_limit"] / peak) if peak else 1.
        events = [{"start_s": round(event["start_s"] * sr) / sr,
                   "end_s": round(event["end_s"] * sr) / sr} for event in source.events]
        save(f"original_{source.source_id}", x * gain, events, [source], {
            "kind": f"original_{source.role}", "source_offset_s": 0.,
            "common_output_gain": gain, "original_events": source.events,
        })

    # 原始统一格式版本照常保留；所有来源类别都补EQ版本，包括困难负例。
    if eq_settings["enabled"]:
        for source in sources:
            for variant in range(1, eq_settings["source_variants"] + 1):
                taps, equalizer = sample_equalizer(cfg, eq_rng)
                x = apply_equalizer(source.x[:frames], taps)
                peak = float(np.max(np.abs(x)))
                gain = min(1., cfg["peak_limit"] / peak) if peak else 1.
                events = [{"start_s": round(event["start_s"] * sr) / sr,
                           "end_s": round(event["end_s"] * sr) / sr} for event in source.events]
                save(f"eq_{source.source_id}_{variant:03d}", x * gain, events, [source], {
                    "kind": f"equalized_source_{source.role}", "source_offset_s": 0.,
                    "source_variant": variant, "common_output_gain": gain,
                    "original_events": source.events, "random_eq": equalizer,
                })

    for foreground, index, clip, active_left, active_right in clips:
        for background in backgrounds:
            variant = 0
            for snr_db in cfg["snr_db"]:
                for _ in range(cfg["repeats_per_snr"]):
                    variant += 1
                    background_start = int(rng.integers(0, len(background.x) - frames + 1))
                    bg = background.x[background_start:background_start + frames]
                    position = int(rng.integers(0, frames - len(clip) + 1))
                    gain_db = float(rng.uniform(*cfg["output_gain_db"]))
                    taps, equalizer = sample_equalizer(cfg, eq_rng)
                    positive, paired_negative, settings = mix_at_snr(
                        bg, clip, position, active_left, active_right,
                        snr_db, cfg["peak_limit"], gain_db, eq_taps=taps,
                    )
                    events = [{"start_s": (position + active_left) / sr,
                               "end_s": (position + active_right) / sr}]
                    pair_id = f"{foreground.source_id}_e{index + 1:02d}_{background.source_id}_{variant:03d}"
                    details = {
                        "pair_id": pair_id, "background_source_id": background.source_id,
                        "foreground_source_id": foreground.source_id, "source_event_index": index,
                        "source_event": foreground.events[index],
                        "background_offset_s": background_start / sr,
                        "foreground_context_position_s": position / sr,
                        "context_s_requested": cfg["context_s"],
                        "fade_s_requested": cfg["fade_s"], "random_eq": equalizer, **settings,
                    }
                    save(f"pos_{pair_id}", positive, events, [foreground, background],
                         {**details, "kind": "positive_mixture"})
                    save(f"neg_{pair_id}", paired_negative, [], [foreground, background],
                         {**details, "kind": "paired_background_only",
                          "foreground_inserted": False})
            print(f"Mixed {foreground.source_id} event {index + 1} + {background.source_id}: {variant} pairs", flush=True)

    # 清单只在全部WAV写出成功后发布，避免失败的半成品进入训练。
    for name, rows in (("train.jsonl", records), ("augmentation_log.jsonl", logs)):
        with (output / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    with (output / "quality.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["recording_id", "kind", "bolt_detected", "duration_s",
                                                   "rms", "peak", "near_full_scale_fraction"])
        writer.writeheader()
        for row in logs:
            writer.writerow({"recording_id": row["recording_id"], "kind": row["augmentation"]["kind"],
                             "bolt_detected": bool(row["events"]), "duration_s": row["duration_s"],
                             **{key: row["output_quality"][key]
                                for key in ("rms", "peak", "near_full_scale_fraction")}})
    summary = {
        "status": "complete", "version": VERSION, "seed": cfg["seed"],
        "sample_rate": sr, "duration_s": frames / sr, "channels": 1, "subtype": "PCM_16",
        "source_foreground_recordings": len(foregrounds),
        "source_events": len(clips), "source_background_recordings": len(backgrounds),
        "source_hard_negative_recordings": sum(source.role == "hard_negative" for source in sources),
        "total_recordings": len(records),
        "positive_recordings": sum(bool(row["events"]) for row in records),
        "negative_recordings": sum(not row["events"] for row in records),
        "mixed_positive_recordings": sum(row["synthetic"] for row in records),
        "paired_negative_recordings": sum(row["augmentation"]["kind"] == "paired_background_only" for row in logs),
        "random_eq": eq_settings,
        "random_eq_augmented_recordings": sum(row["augmentation"]["random_eq"]["applied"] for row in logs),
        "equalized_source_recordings": sum(row["augmentation"]["kind"].startswith("equalized_source_") for row in logs),
        "source_warnings": [{"id": source.source_id, "warnings": source.quality["warnings"]}
                            for source in sources if source.quality["warnings"]],
        "maximum_output_peak": max(row["output_quality"]["peak"] for row in logs),
        "output_near_full_scale_samples": any(row["output_quality"]["near_full_scale_fraction"] > 0 for row in logs),
        "manifest": str(output / "train.jsonl"), "processing_seconds": time.perf_counter() - started,
        "filename_scheme": "sequential_decimal_minimum_2_digits",
        "waveform_images": len(records), "spectrogram_images": len(records), "plot_spec": PLOT_SPEC,
        "limitations": [
            "Synthetic variants do not increase the number of independent real bolt-drop events.",
            "All sources and derivatives are in one train group; no validation or test data generated.",
            "Source clipping and residual foreground noise remain; mixing approximates operating conditions.",
            "Time labels come from reviewed source markers; generated files are not individually listened.",
            "EQ preserves duration and compensates FIR delay; short filter spreading around transients remains.",
        ],
    }
    write_json(output / "summary.json", summary)
    (output / "RUNNING.json").unlink()
    return summary


def install_renumbered_files(root: Path, stage: Path, backup: Path) -> None:
    """保留目录本身，逐文件原子替换；适用于Windows中不能移动的打开目录。"""
    root, stage, backup = root.resolve(strict=True), stage.resolve(strict=True), backup.resolve()
    if (not stage.is_relative_to(root.parent) or not backup.is_relative_to(root.parent)
            or not stage.name.startswith(f".{root.name}_renumber_")
            or backup.parent != root.parent / "_backups"):
        raise ValueError("迁移目录/备份路径超出预期范围")
    if any(path.is_symlink() or path.is_junction() for path in stage.rglob("*")):
        raise ValueError("临时副本包含链接，拒绝迁移")
    with (stage / "renaming_map.csv").open(encoding="utf-8-sig", newline="") as stream:
        mapping = list(csv.DictReader(stream))
    for row in mapping:
        for key in ("old_audio", "new_audio", "old_labels", "new_labels"):
            value = row[key]
            if value and (Path(value).is_absolute() or not (root / value).resolve().is_relative_to(root)):
                raise ValueError("迁移映射包含越界路径")
        if sha256(root / row["old_audio"]) != row["wav_sha256"] or sha256(stage / row["new_audio"]) != row["wav_sha256"]:
            raise ValueError("原音频或临时副本在迁移期间发生变化")
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(root, backup)
    marker = root / "RENUMBERING.json"
    write_json(marker, {"status": "incomplete", "backup": str(backup), "stage": str(stage)})
    installed, removed = [], []
    token = uuid4().hex

    def replace_from(source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{token}.part")
        try:
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    prepared = [path for path in stage.rglob("*") if path.is_file()]
    data_directories = {"audio", "labels", "waveforms", "spectrograms"}
    prepared.sort(key=lambda path: (path.relative_to(stage).parts[0] not in data_directories,
                                    path.name == "summary.json", str(path)))
    targets = {path.relative_to(stage) for path in prepared}
    try:
        for source in prepared:
            relative = source.relative_to(stage)
            replace_from(source, root / relative)
            installed.append(relative)
        for row in mapping:
            for key in ("old_audio", "old_labels"):
                old = Path(row[key]) if row[key] else None
                if old is not None and old not in targets:
                    (root / old).unlink()
                    removed.append(old)
    except OSError as exc:
        # 从完整备份回滚已写入/移除的文件，保留临时副本供定位问题。
        try:
            for relative in removed:
                replace_from(backup / relative, root / relative)
            for relative in reversed(installed):
                if (backup / relative).exists():
                    replace_from(backup / relative, root / relative)
                elif (root / relative).exists():
                    (root / relative).unlink()
            marker.unlink()
        except OSError as rollback_error:
            raise OSError(f"迁移失败且回滚未完成；恢复备份: {backup}; {rollback_error}") from exc
        raise OSError(f"迁移未完成，已回滚；原结果备份: {backup}; {exc}") from exc
    marker.unlink()
    # stage已验证为当前数据集父目录中的专用临时目录，原文件另有完整备份。
    try:
        shutil.rmtree(stage)
    except OSError:
        print(f"Warning: migrated successfully; temporary copy remains at {stage}", file=sys.stderr)


def renumber_existing(root: Path) -> dict:
    """在副本中迁移既有数据、补图，核对WAV哈希后替换，原目录保留为备份。"""
    started = time.perf_counter()
    root = root.resolve(strict=True)
    if any((root / name).exists() for name in ("RUNNING.json", "RENUMBERING.json", ".git")):
        raise ValueError("请选择已生成完成的数据集目录，而不是项目根目录或半成品目录")
    if any(path.is_symlink() or path.is_junction() for path in root.rglob("*")):
        raise ValueError("数据集包含符号链接/目录联接，不能安全迁移")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8-sig"))
    if summary.get("status") != "complete":
        raise ValueError("原数据集尚未完成，不能迁移")
    records = [json.loads(line) for line in (root / "train.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    log_rows = [json.loads(line) for line in (root / "augmentation_log.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    old_ids = [row["recording_id"] for row in records]
    logs = {row["recording_id"]: row for row in log_rows}
    if not records or len(set(old_ids)) != len(records) or set(old_ids) != set(logs) or len(log_rows) != len(records):
        raise ValueError("训练清单和日志的样本ID不一致或存在重复")
    with (root / "quality.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        quality_columns = reader.fieldnames
        quality_rows = list(reader)
    if len(quality_rows) != len(records) or {row["recording_id"] for row in quality_rows} != set(old_ids):
        raise ValueError("质量表和训练清单不一致")

    def inside(relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or not (root / path).resolve().is_relative_to(root):
            raise ValueError(f"文件引用超出数据集目录: {relative}")
        return root / path

    mapping, moves = [], []
    current = True
    for index, row in enumerate(records, 1):
        new_id = f"{index:02d}"
        new_audio = f"audio/{new_id}.wav"
        new_labels = f"labels/{new_id}.csv" if row.get("labels") else None
        audio = inside(row["audio"])
        if not audio.is_file() or logs[row["recording_id"]]["audio"] != row["audio"]:
            raise ValueError(f"清单与音频/日志不一致: {row['recording_id']}")
        moves.append((row["audio"], new_audio))
        if row.get("labels"):
            if not inside(row["labels"]).is_file():
                raise ValueError(f"标记文件不存在: {row['labels']}")
            moves.append((row["labels"], new_labels))
        plots = {"waveform": f"waveforms/{new_id}.png", "spectrogram": f"spectrograms/{new_id}.png"}
        current &= (row["recording_id"] == new_id and row["audio"] == new_audio
                    and row.get("labels") == new_labels and row.get("plots") == plots
                    and all(inside(path).is_file() for path in plots.values()))
        mapping.append({"old_recording_id": row["recording_id"], "new_recording_id": new_id,
                        "old_audio": row["audio"], "new_audio": new_audio,
                        "old_labels": row.get("labels") or "", "new_labels": new_labels or "",
                        "wav_sha256": sha256(audio)})
    if current:
        return {"status": "unchanged", "recordings": len(records), "directory": str(root)}
    previous_paths = {old for old, _ in moves}
    if len(previous_paths) != len(moves):
        raise ValueError("多条样本引用同一个文件")
    for old, new in moves:
        if inside(new).exists() and new not in previous_paths:
            raise FileExistsError(f"目标文件与清单外文件冲突: {new}")
    if {path.resolve() for path in (root / "audio").rglob("*.wav")} != {
        inside(row["audio"]).resolve() for row in records
    }:
        raise ValueError("audio目录中有未列入清单的WAV，请先确认它们的来源")

    token = uuid4().hex[:12]
    stage = root.parent / f".{root.name}_renumber_{token}"
    backup = root.parent / "_backups" / f"{root.name}_before_renumber_{token}"
    # Windows目录移动前校验最终绝对路径；不递归删除任何原始目录。
    if not stage.resolve().is_relative_to(root.parent) or not backup.resolve().is_relative_to(root.parent):
        raise ValueError("临时/备份路径超出指定数据集的父目录")
    shutil.copytree(root, stage)
    temporary_moves = []
    for index, (old, new) in enumerate(moves):
        temporary = stage / f".rename_{token}_{index}{Path(old).suffix}"
        (stage / old).rename(temporary)
        temporary_moves.append((temporary, stage / new))
    for temporary, destination in temporary_moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)

    for index, (row, entry) in enumerate(zip(records, mapping), 1):
        old_id, new_id = entry["old_recording_id"], entry["new_recording_id"]
        plots = {"waveform": f"waveforms/{new_id}.png", "spectrogram": f"spectrograms/{new_id}.png"}
        if not all((stage / path).is_file() for path in plots.values()):
            for path in plots.values():
                target = stage / path
                if target.exists():
                    target.unlink()  # 仅移除临时副本中待补齐的该样本图片。
            render_audio_plots(stage / entry["new_audio"], stage, new_id)
        replacement = {"recording_id": new_id, "audio": entry["new_audio"],
                       "labels": entry["new_labels"] or None, "plots": plots,
                       "derivation_id": row.get("derivation_id", old_id)}
        row.update(replacement)
        logs[old_id].update(replacement)
        if sha256(stage / entry["new_audio"]) != entry["wav_sha256"]:
            raise ValueError(f"迁移后WAV内容发生变化，原目录仍保留: {old_id}")
        if index % 10 == 0 or index == len(records):
            print(f"Renumbered and plotted {index}/{len(records)}", flush=True)
    for name, rows in (("train.jsonl", records), ("augmentation_log.jsonl", log_rows)):
        (stage / name).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    id_mapping = {row["old_recording_id"]: row["new_recording_id"] for row in mapping}
    with (stage / "quality.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=quality_columns)
        writer.writeheader()
        for row in quality_rows:
            writer.writerow({**row, "recording_id": id_mapping[row["recording_id"]]})
    with (stage / "renaming_map.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    summary.update({"manifest": str(root / "train.jsonl"),
                    "filename_scheme": "sequential_decimal_minimum_2_digits",
                    "waveform_images": len(records), "spectrogram_images": len(records),
                    "plot_spec": PLOT_SPEC, "renumber_backup": str(backup),
                    "renumber_processing_seconds": time.perf_counter() - started})
    write_json(stage / "summary.json", summary)
    install_renumbered_files(root, stage, backup)
    return {"status": "complete", "recordings": len(records), "directory": str(root),
            "backup": str(backup), "waveform_images": len(records), "spectrogram_images": len(records),
            "audio_content_changed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="扫描WAV元信息及质量，不根据文件名猜标签")
    scan.add_argument("--input", type=Path, required=True)
    scan.add_argument("--output", type=Path, required=True)
    make = sub.add_parser("generate", help="按已确认的来源配置生成训练增强集")
    make.add_argument("--config", type=Path, default=Path(__file__).with_name("augmentation.json"))
    make.add_argument("--output", type=Path, required=True, help="必须是不存在的新目录")
    rename = sub.add_parser("renumber", help="将既有完整数据集改为连续编号，补齐两类图片并保留备份")
    rename.add_argument("--input", type=Path, required=True, help="包含train.jsonl和summary.json的数据集目录")
    args = parser.parse_args()
    try:
        if args.command == "scan":
            result = inventory(args.input.resolve(strict=True), args.output)
        elif args.command == "renumber":
            result = renumber_existing(args.input)
        else:
            result = generate(args.config, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, sf.LibsndfileError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
