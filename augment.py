"""螺栓声音数据增强：真实背景混合、时间标签、训练清单和来源审计。

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
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

VERSION = 1
AUDITION_COLUMNS = ["Name", "Start", "Duration", "Time Format", "Type", "Description"]


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


def load_config(path: Path) -> tuple[dict, list[Source]]:
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if cfg.get("version") != VERSION:
        raise ValueError("不支持的配置版本")
    sr = cfg["sample_rate"]
    if type(sr) is not int or not 1000 <= sr <= 192000:
        raise ValueError("sample_rate必须为1000～192000的整数")
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
               peak_limit: float, gain_db: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """事件标注区间内计算功率；正例与配对负例应用完全相同的全局增益。"""
    if not 0 <= start <= len(background) - len(foreground):
        raise ValueError("事件及其上下文放不进背景")
    if not 0 <= active_start < active_end <= len(foreground):
        raise ValueError("无效事件活动区间")
    active = slice(start + active_start, start + active_end)
    signal_power = float(np.mean(np.square(foreground[active_start:active_end])))
    noise_power = float(np.mean(np.square(background[active])))
    if min(signal_power, noise_power) <= 1e-16:
        raise ValueError("事件或对应背景接近静音，不能按指定信噪比混合")
    multiplier = math.sqrt(noise_power / signal_power) * 10 ** (snr_db / 20)
    event = np.zeros_like(background)
    event[start:start + len(foreground)] = foreground * multiplier
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
    }


def save_pcm(path: Path, x: np.ndarray, sr: int) -> dict:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖: {path}")
    if not np.isfinite(x).all() or np.max(np.abs(x)) >= .999:
        raise ValueError("输出含非有限值或接近量程边界，拒绝写出")
    sf.write(path, x, sr, subtype="PCM_16")
    written, _ = sf.read(path, dtype="float64")
    return signal_quality(written)


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
    records, logs = [], []

    def save(recording_id: str, x: np.ndarray, events: list[dict], used: list[Source],
             details: dict) -> None:
        relative_audio = f"audio/{recording_id}.wav"
        quality = save_pcm(output / relative_audio, x, sr)
        labels = None
        if events:
            labels = f"labels/{recording_id}.csv"
            export_markers(output / labels, events)
        row = {
            "recording_id": recording_id, "audio": relative_audio,
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
                    positive, paired_negative, settings = mix_at_snr(
                        bg, clip, position, active_left, active_right,
                        snr_db, cfg["peak_limit"], gain_db,
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
                        "fade_s_requested": cfg["fade_s"], **settings,
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
        "source_warnings": [{"id": source.source_id, "warnings": source.quality["warnings"]}
                            for source in sources if source.quality["warnings"]],
        "maximum_output_peak": max(row["output_quality"]["peak"] for row in logs),
        "output_near_full_scale_samples": any(row["output_quality"]["near_full_scale_fraction"] > 0 for row in logs),
        "manifest": str(output / "train.jsonl"), "processing_seconds": time.perf_counter() - started,
        "limitations": [
            "Synthetic variants do not increase the number of independent real bolt-drop events.",
            "All sources and derivatives are in one train group; no validation or test data generated.",
            "Source clipping and residual foreground noise remain; mixing approximates operating conditions.",
            "Time labels come from reviewed source markers; generated files are not individually listened.",
        ],
    }
    write_json(output / "summary.json", summary)
    (output / "RUNNING.json").unlink()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan", help="扫描WAV元信息及质量，不根据文件名猜标签")
    scan.add_argument("--input", type=Path, required=True)
    scan.add_argument("--output", type=Path, required=True)
    make = sub.add_parser("generate", help="按已确认的来源配置生成训练增强集")
    make.add_argument("--config", type=Path, default=Path(__file__).with_name("augmentation.json"))
    make.add_argument("--output", type=Path, required=True, help="必须是不存在的新目录")
    args = parser.parse_args()
    try:
        if args.command == "scan":
            result = inventory(args.input.resolve(strict=True), args.output)
        else:
            result = generate(args.config, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, sf.LibsndfileError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
