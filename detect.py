"""
轮毂音频 V3：先按采集时间重命名，再按波形形态分为“平稳型 / 非平稳型”。

依赖：python -m pip install numpy scipy pandas matplotlib
运行：python detect.py --input "D:/hub_wav" --output "D:/hub_result_v3" --rename-mode copy

默认行为：
  1. 修改输入 WAV 文件名，例如 *_20261018003333_*.wav -> 2026_10_18_00_33.wav。
  2. 同一分钟重名时增加 _002、_003；不能解析时间时保留原名。
  3. 全部可读取音频参与二分类，包括全静音、部分静音和满量程文件。
     持续满量程且形态稳定仍为平稳型，不按绝对幅值/满量程比例判异常。
  4. 保存类别音频副本、结果表、逐帧特征、候选波动区间和分析图。

--rename-mode copy 会先复制并重命名到输出目录 renamed，不修改原文件。
--rename-mode none 跳过重命名；--no-copy 不生成分类音频副本。
输出目录必须为空或不存在，避免与旧结果混淆。

这是波形形态筛查，不是设备故障诊断，也不是严格的统计平稳性检验。
不按档位或跨文件绝对幅值分组；阈值是需要真实样本复核的工程初值。
V3 增加 10 ms 局部变化检测，以及“全段 RMS + 峰值”双限制低能量门控。
门控默认值按数字满量程设定，不是设备标准；可用 --quiet-reference 指定
一条已确认停机的 WAV，用其 RMS 与峰值的 3 倍校准门控（仍需人工复核）。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import warnings

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.ndimage import median_filter

VERSION = 3
DEFAULTS = {
    'frame_ms': 100.0, 'hop_ms': 25.0, 'smooth_ms': 250.0,
    'silence_peak': 1e-7, 'channel': 0,
    'cf_threshold': 6.0, 'kurt_threshold': 10.0,
    'strong_cf': 12.0, 'strong_kurt': 40.0,
    'burst_ratio': 3.5, 'envelope_ratio': 1.8,
    'max_clip_fraction': 0.001,
    # 低能量门控：必须同时满足 RMS 和峰值上限，才按低能量平稳处理。
    # 可将 quiet_rms 设为 0 禁用。非零默认值只适用于当前数字量程假设，
    # 应优先通过确认停机录音校准；低于此门控的真实弱事件可能被忽略。
    'quiet_rms': 3e-4, 'quiet_peak': 3e-3,
    # 短窗口与局部背景：比全段 CF 更敏感于高背景中的短暂相对变化。
    'short_frame_ms': 10.0, 'local_window_ms': 1000.0,
    'local_z': 6.0, 'local_rms_ratio': 1.4, 'local_peak_ratio': 1.4,
}


def time_name(filename):
    """从文件名提取时间，精确到分钟；返回 (目标 stem, 时间说明)。

    优先识别已改名的 YYYY_MM_DD_HH_MM[_编号]，保证重复运行不再加后缀。
    对原始名称仅接受独立的 12/14 位数字字段，并验证日期合法性。
    若有多个有效时间字段，不猜测含义，保留原名并记录说明。
    """
    stem = Path(filename).stem
    match = re.fullmatch(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2})(?:_\d{3,})?', stem)
    if match:
        try:
            dt = datetime.strptime(match[1], '%Y_%m_%d_%H_%M')
            return stem, dt.isoformat(timespec='minutes')
        except ValueError:
            return None, '日期不合法，保留原名'
    candidates = []
    for token in re.findall(r'(?<![A-Za-z0-9])(\d{14}|\d{12})(?![A-Za-z0-9])', stem):
        try:
            dt = datetime.strptime(token, '%Y%m%d%H%M%S' if len(token) == 14 else '%Y%m%d%H%M')
            candidates.append(dt)
        except ValueError:
            pass
    if len(candidates) != 1:
        return None, '未找到唯一合法时间字段，保留原名'
    dt = candidates[0]
    return dt.strftime('%Y_%m_%d_%H_%M'), dt.isoformat(timespec='seconds')


def rename_no_overwrite(source, target):
    """同目录改名，不覆盖已有文件。

    Windows 的 rename 在目标存在时会失败；其他系统采用独占创建硬链接
    后删除旧目录项，保留同一份文件内容。失败时原路径通常仍可用。
    若链接已创建但删除旧名失败，回报错误，不擅自删除任何目录项。
    """
    if os.name == 'nt':
        os.rename(source, target)
    else:
        os.link(source, target)  # 原子地拒绝已存在目标，不复制文件内容。
        source.unlink()


def prepare_files(root, output, mode):
    """完整枚举后逐个改名，返回原名/新名/实际分析路径映射。

    使用 JSONL 逐条记录计划和完成状态并 flush，方便中途失败时查找原名。
    rename_map.csv 是运行结束后便于 Excel 查看的一份完整映射。
    """
    paths = sorted(p for p in root.rglob('*')
                   if p.is_file() and p.suffix.lower() == '.wav'
                   and not p.resolve().is_relative_to(output))
    if not paths:
        raise ValueError('输入目录中没有 WAV 文件。')
    entries, reserved = [], set()
    with (output / 'rename_journal.jsonl').open('w', encoding='utf-8') as journal:
        for index, source in enumerate(paths):
            rel = source.relative_to(root)
            stem, time_note = time_name(source.name)
            base = source.parent if mode != 'copy' else output / 'renamed' / rel.parent
            base.mkdir(parents=True, exist_ok=True)
            target = source if mode == 'none' else base / ((stem or source.stem) + '.wav')
            # 同一分钟的录音不能丢弃或覆盖；在同一目录中追加编号。
            if target != source:
                original_stem = target.stem
                number = 2
                while target.exists() or target in reserved:
                    target = base / f'{original_stem}_{number:03d}.wav'
                    number += 1
            reserved.add(target)
            entry = {
                'frame_id': f'{index:05d}', 'original_file': rel.as_posix(),
                'file': target.relative_to(output / 'renamed' if mode == 'copy' else root).as_posix(),
                'time_note': time_note, 'rename_status': 'unchanged',
                'rename_error': '', 'source_path': str(source), 'analysis_path': str(target),
            }
            journal.write(json.dumps({'phase': 'planned', **entry}, ensure_ascii=False) + '\n')
            journal.flush()
            try:
                if mode == 'copy':
                    # xb 独占创建目标，避免目标在规划后出现时被覆盖。
                    with source.open('rb') as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
                    entry['rename_status'] = 'copied'
                elif mode == 'inplace' and source != target:
                    rename_no_overwrite(source, target)
                    entry['rename_status'] = 'renamed'
            except OSError as exc:
                # 改名失败仍分析原始文件，不遗漏可读取的数据。
                entry.update(rename_status='failed', rename_error=str(exc),
                             file=rel.as_posix(), analysis_path=str(source))
            journal.write(json.dumps({'phase': 'finished', **entry}, ensure_ascii=False) + '\n')
            journal.flush()
            entries.append(entry)
    pd.DataFrame(entries).to_csv(output / 'rename_map.csv', index=False, encoding='utf-8-sig')
    return entries


def read_signal(path, cfg):
    """读取指定声道，按数字满量程缩放并去直流；不因静音或削顶而拒绝。

    不混合多声道，避免反相抵消。WAV 元数据块警告保留在质量提示中；
    真正的读取错误、空数组、非有限样本仍不能可靠分类，交给 errors.csv。
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', wavfile.WavFileWarning)
        sr, raw = wavfile.read(path)
    notes = [str(w.message) for w in caught]
    if raw.ndim not in (1, 2) or len(raw) == 0 or sr <= 0:
        raise ValueError('空音频或无效采样率/维度')
    if raw.ndim == 2:
        if cfg['channel'] >= raw.shape[1]:
            raise ValueError('指定声道不存在')
        raw = raw[:, cfg['channel']]
    elif cfg['channel'] != 0:
        raise ValueError('单声道文件只能使用 --channel 0')
    if raw.dtype.kind == 'u':
        scale = (np.iinfo(raw.dtype).max + 1) / 2
        x = (raw.astype(float) - scale) / scale
    elif raw.dtype.kind == 'i':
        limits = np.iinfo(raw.dtype)
        x = raw.astype(float) / max(abs(limits.min), limits.max)
    elif raw.dtype.kind == 'f':
        x = raw.astype(float)
    else:
        raise ValueError('不支持的 WAV 样本格式')
    if not np.isfinite(x).all():
        raise ValueError('含 NaN 或无穷值，不能可靠分类')
    clip_fraction = float(np.mean(np.abs(x) >= .999))
    original_mean = float(x.mean())
    # 原始均值用于区分真正静音与非零常量；两者都可归为平稳型，
    # 但不能把满量程常量去均值后得到的 0 误记成停机静音。
    metadata = {'original_mean': original_mean,
                'original_peak': float(np.max(np.abs(x))),
                'original_rms': float(np.sqrt(np.mean(x*x)))}
    x -= original_mean
    if clip_fraction > cfg['max_clip_fraction']:
        notes.append('近满量程样本比例较高；仅作记录，不作为非平稳依据')
    if len(x) < max(2, round(sr * cfg['frame_ms'] / 1000)):
        notes.append('音频短于标准窗口，使用实际长度；结果可靠性较低')
    if abs(original_mean) > .1:
        notes.append('直流偏置较大，已去均值；请检查采集')
    return sr, x, clip_fraction, '; '.join(notes), metadata


def local_change_features(x, sr, frames, cfg):
    """补充短窗口局部变化分数，再映射到主分析帧。

    以不重叠的 10 ms 小块计算 RMS/峰值/CF，与附近约 1 秒的中位数比较。
    对数残差的滚动 MAD 提供背景波动尺度，尺度下限 0.05 避免过敏。
    局部 RMS 升高需同时满足倍率和稳健 z 门槛；局部峰值升高还需
    RMS 至少上升 15% 或 CF 至少上升 30%，降低随机最大值导致的误报。
    所有这些门槛是可复核的工程规则，不是概率，也不保证故障召回率。
    """
    block = max(2, round(sr * cfg['short_frame_ms'] / 1000))
    starts = np.arange(0, len(x), block)
    counts = np.diff(np.append(starts, len(x)))
    rms = np.sqrt(np.maximum(0, np.add.reduceat(x*x, starts) / counts))
    peak = np.maximum.reduceat(np.abs(x), starts)
    floor = cfg['silence_peak']
    cf = peak / np.maximum(rms, floor)
    # 只对短块序列滤波，不对原采样率信号执行大型中值滤波。
    width = max(3, round(cfg['local_window_ms'] / cfg['short_frame_ms']))
    width = min(width, max(1, len(starts)))
    if width % 2 == 0:
        width = max(1, width - 1)
    output = []
    for values in (rms, peak):
        logs = np.log(np.maximum(values, floor))
        background = median_filter(logs, size=width, mode='reflect')
        residual = logs - background
        scale = np.maximum(1.4826 * median_filter(np.abs(residual), size=width, mode='reflect'), .05)
        output.append((np.exp(np.clip(residual, -100, 100)), np.maximum(residual, 0) / scale))
    (rms_ratio, rms_z), (peak_ratio, peak_z) = output
    baseline_cf = median_filter(cf, size=width, mode='reflect')
    crest_ratio = cf / np.maximum(baseline_cf, 1.0)
    # 分数达到 1 代表同时满足条件；单纯幅值一直很高不会触发。
    rms_score = np.minimum(rms_ratio / cfg['local_rms_ratio'], rms_z / cfg['local_z'])
    corroboration = np.maximum(rms_ratio / 1.15, crest_ratio / 1.3)
    peak_score = np.minimum.reduce([peak_ratio / cfg['local_peak_ratio'],
                                     peak_z / cfg['local_z'], corroboration])
    local = np.maximum(rms_score, peak_score)
    local[(rms <= floor) | (counts < block)] = 0  # 不完整尾块不作局部突变依据。
    # 将所有与主帧重叠的小块最大分数合并，便于沿用既有 frames/events 输出。
    ends_s = (starts + counts) / sr
    starts_s = starts / sr
    mapped, mapped_rms, mapped_peak = [], [], []
    for frame in frames.itertuples():
        left = int(np.searchsorted(ends_s, frame.start_s, side='right'))
        right = int(np.searchsorted(starts_s, frame.end_s, side='left'))
        mapped.append(float(np.max(local[left:right])) if right > left else 0.0)
        mapped_rms.append(float(np.max(rms_ratio[left:right])) if right > left else 1.0)
        mapped_peak.append(float(np.max(peak_ratio[left:right])) if right > left else 1.0)
    frames['local_score'] = mapped
    frames['local_rms_ratio'] = mapped_rms
    frames['local_peak_ratio'] = mapped_peak
    return float(max(mapped, default=0.0))


def classify(path, cfg):
    """逐条音频独立判断，不利用不同文件间的绝对 RMS 大小分类。

    四类形态证据：
      冲击：帧内峰值因子和峭度联合升高，或其中一项极强。
      突增：帧 RMS 相对本文件典型 RMS 的倍率升高。
      波动：平滑 RMS 包络的 P95/P05 倍率升高，覆盖明显升降/间歇。
      局部：10 ms 窗口与附近背景比较，避免高背景稀释短暂变化。
    全段接近零的波形直接归平稳型；部分静音仍需检查非静音区和包络。
    """
    sr, x, clipping, quality, metadata = read_signal(path, cfg)
    floor = cfg['silence_peak']
    peak = float(np.max(np.abs(x)))
    flat = peak <= floor
    silent = flat and metadata['original_peak'] <= floor
    global_rms = float(np.sqrt(np.mean(x*x)))
    # 必须同时低 RMS、低峰值，才把很弱的底噪视为平稳；
    # 单凭平均能量低，不能屏蔽静音背景中的较大孤立冲击。
    quiet_gate = bool(cfg['quiet_rms'] > 0 and global_rms <= cfg['quiet_rms']
                      and peak <= cfg['quiet_peak'])
    n = min(len(x), max(2, round(sr * cfg['frame_ms'] / 1000)))
    hop = max(1, round(sr * cfg['hop_ms'] / 1000))
    starts = list(range(0, len(x) - n + 1, hop))
    if starts[-1] != len(x) - n:
        starts.append(len(x) - n)
    rows = []
    for start in starts:
        y = x[start:start+n]
        rms = float(np.sqrt(np.mean(y*y)))
        centered = y - y.mean()
        variance = float(np.mean(centered*centered))
        # 静音/常数帧 CF、峭度记为 0，不作无定义的除法。
        cf = float(np.max(np.abs(y)) / rms) if rms > floor else 0.0
        kurt = float(np.mean(centered**4) / variance**2) if variance > floor**2 else 0.0
        rows.append((start/sr, (start+n)/sr, rms, cf, kurt))
    f = pd.DataFrame(rows, columns=['start_s', 'end_s', 'rms', 'crest_factor', 'kurtosis'])
    # 用本文件整体的典型能量比较突增，不使用其他档位的幅值。
    # 大部分时间静音时保留检测孤立短脉冲的能力；以活动帧中位数的
    # 5% 设置相对下限，避免零分母以及不必要的百万倍分数。
    active = f.loc[f.rms > floor, 'rms']
    active_typical = float(active.median()) if len(active) else floor
    reference = max(float(f.rms.median()), active_typical * .05, floor)
    f['rms_ratio'] = f.rms / reference
    # 中值平滑降低随机帧间起伏的影响；不会改变冲击判据使用的原始帧。
    smooth_frames = max(1, round(cfg['smooth_ms'] / cfg['hop_ms']))
    f['rms_smooth'] = f.rms.rolling(smooth_frames, center=True, min_periods=1).median()
    p05, p95 = np.percentile(f.rms_smooth, [5, 95])
    envelope_ratio = float(p95 / max(p05, active_typical * .05, floor)) if p95 > floor else 1.0
    # CF 与峭度相关，要求中等强度时两者都高，或单个特征达到强门槛。
    f['shape_score'] = np.maximum.reduce([
        np.minimum(f.crest_factor / cfg['cf_threshold'], f['kurtosis'] / cfg['kurt_threshold']),
        f.crest_factor / cfg['strong_cf'], f['kurtosis'] / cfg['strong_kurt'],
    ])
    f['burst_score'] = f.rms_ratio / cfg['burst_ratio']
    shape_score = float(f.shape_score.max())
    burst_score = float(f.burst_score.max())
    fluctuation_score = envelope_ratio / cfg['envelope_ratio']
    local_score = local_change_features(x, sr, f, cfg)
    raw_score = max(shape_score, burst_score, fluctuation_score, local_score)
    score = raw_score if not (flat or quiet_gate) else 0.0
    nonstationary = score >= 1
    reasons = []
    if silent:
        reasons.append('全段近乎静音/平直，按要求归入平稳型（可能停机）')
    elif flat:
        reasons.append('全段为非零平直常量，按形态归入平稳型；不推断是否停机')
    elif quiet_gate:
        reasons.append('RMS 与峰值均低于低能量门控，按停机/底噪候选归入平稳型；不能仅凭音频确认停机')
    else:
        if shape_score >= 1:
            reasons.append('存在尖峰/冲击：峰值因子或峭度偏高')
        if burst_score >= 1:
            reasons.append('局部 RMS 相对本文件典型水平明显突增')
        if fluctuation_score >= 1:
            reasons.append('平滑 RMS 包络存在明显升降或间歇')
        if local_score >= 1:
            reasons.append('短窗口相对局部背景出现明显尖峰或能量突增')
        if not reasons:
            reasons.append('未检测到超过当前阈值的尖峰或明显能量波动')
    if clipping > cfg['max_clip_fraction'] and not nonstationary:
        reasons.append('存在较多满量程样本，但未发现超过阈值的形态波动，归入平稳型')
    # 定位冲击/突增窗口；若包络波动判据触发，标记偏离本文件包络中位数
    # 超过 sqrt(envelope_ratio门槛) 的窗口，兼顾上升和下降。
    typical = max(float(f.rms_smooth.median()), floor)
    factor = math.sqrt(cfg['envelope_ratio'])
    envelope_mask = ((f.rms_smooth >= typical*factor) | (f.rms_smooth <= typical/factor))
    f['variation_candidate'] = ((f.shape_score >= 1) | (f.burst_score >= 1) | (f.local_score >= 1)
                               | ((fluctuation_score >= 1) & envelope_mask)) & (not (flat or quiet_gate))
    row = {
        'label': '非平稳型' if nonstationary else '平稳型', 'score': float(score),
        'reason': '; '.join(reasons), 'is_silent': bool(silent), 'is_flat': bool(flat),
        'quiet_gate': quiet_gate, 'raw_score': float(raw_score), 'local_score': local_score,
        'rms_dbfs': float(20*np.log10(max(global_rms, 1e-15))),
        'peak_dbfs': float(20*np.log10(max(peak, 1e-15))),
        'has_full_scale_samples': bool(clipping > cfg['max_clip_fraction']),
        **metadata,
        'duration_s': len(x)/sr, 'sample_rate': int(sr), 'n_frames': len(f),
        'absolute_peak': peak, 'rms_global': global_rms,
        'cf_max': float(f.crest_factor.max()), 'kurt_max': float(f['kurtosis'].max()),
        'rms_reference': reference, 'rms_ratio_max': float(f.rms_ratio.max()),
        'rms_smooth_p05': float(p05), 'rms_smooth_p95': float(p95),
        'envelope_ratio': envelope_ratio, 'shape_score': shape_score,
        'burst_score': burst_score, 'fluctuation_score': fluctuation_score,
        'clip_fraction': clipping, 'quality_warning': quality,
    }
    return row, f


def merge_events(f):
    """合并重叠的候选波动窗口；区间边界精度受窗长/步长限制。"""
    intervals = []
    for r in f[f.variation_candidate].itertuples():
        if intervals and r.start_s <= intervals[-1][1] + 1e-9:
            intervals[-1][1] = max(intervals[-1][1], r.end_s)
        else:
            intervals.append([r.start_s, r.end_s])
    return intervals


def plot_file(path, row, f, cfg, destination):
    """保存波形、峰值因子、峭度、RMS、局部变化五行图。

    红色为生效的候选波动；若触发 quiet gate，保留原始局部分数供复核，
    但它们不再触发文件级非平稳标签。标题同时显示数字 RMS 与门控状态。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sr, x, _, _, _ = read_signal(path, cfg)
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True, constrained_layout=True)
    # 分块 min/max 包络保留尖峰，包括不能整除的末尾样本。
    block = max(1, math.ceil(len(x)/20000))
    indices = np.arange(0, len(x), block)
    axes[0].fill_between(indices/sr, np.minimum.reduceat(x, indices),
                         np.maximum.reduceat(x, indices), color='#2563eb')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f"{row['frame_id']} | {'nonstationary' if row['label']=='非平稳型' else 'stable'} | score={row['score']:.2f} | RMS={row['rms_dbfs']:.1f} dBFS | quiet={row['quiet_gate']}")
    for ax, name in zip(axes[1:], ['crest_factor', 'kurtosis', 'rms', 'local_score']):
        ax.plot((f.start_s + f.end_s)/2, f[name], linewidth=.8)
        ax.set_ylabel(name)
    axes[-1].axhline(1, color='red', linestyle='--', linewidth=.8)
    for ax in axes:
        for left, right in merge_events(f):
            ax.axvspan(left, right, alpha=.18, color='red')
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Time (s)')
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def main():
    """先改名，再逐条分类；不因一条文件损坏而停止整批分析。"""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--version', action='version', version='detect.py V3 (local-change + quiet gate)')
    ap.add_argument('--input', type=Path, required=True, help='递归读取的 WAV 目录')
    ap.add_argument('--output', type=Path, default=Path('results_v2'), help='新的/空的输出目录')
    ap.add_argument('--rename-mode', choices=['inplace', 'copy', 'none'], default='inplace',
                    help='默认 inplace 修改原名；copy 只改副本；none 不改名')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--model', type=Path, help='加载 V3 固定阈值，不支持 V1/V2')
    group.add_argument('--quiet-reference', type=Path, help='已确认停机的 WAV，用其 RMS/峰值的 3 倍校准低能量门控')
    ap.add_argument('--no-copy', action='store_true', help='不复制到分类目录，仍保存分类表')
    ap.add_argument('--copy', action='store_true', help='兼容旧命令；V2 默认已经复制分类音频')
    ap.add_argument('--plots', type=int, default=12, help='绘制最高分的 N 条；0 不绘图')
    for key, value in DEFAULTS.items():
        ap.add_argument('--' + key.replace('_', '-'), type=int if key == 'channel' else float,
                        default=value, help=f'默认 {value}，详见代码中的对应特征说明')
    args = ap.parse_args()
    cfg = {key: getattr(args, key) for key in DEFAULTS}
    if args.model:
        model = json.loads(args.model.read_text(encoding='utf-8'))
        if model.get('version') != VERSION:
            ap.error('不能加载旧版模型，请使用 V3 重新运行。')
        cfg = model['config']
    if set(cfg) != set(DEFAULTS) or not all(np.isfinite(v) for v in cfg.values()):
        ap.error('模型配置缺失或含非有限参数。')
    if (not 0 < cfg['hop_ms'] <= cfg['frame_ms'] or cfg['smooth_ms'] <= 0
        or cfg['channel'] < 0 or int(cfg['channel']) != cfg['channel']
        or cfg['silence_peak'] <= 0 or not 0 <= cfg['max_clip_fraction'] <= 1
        or any(cfg[k] <= 1 for k in ['cf_threshold','kurt_threshold','strong_cf','strong_kurt','burst_ratio','envelope_ratio'])
        or cfg['quiet_rms'] < 0 or cfg['quiet_peak'] < 0
        or cfg['short_frame_ms'] <= 0 or cfg['local_window_ms'] < 3*cfg['short_frame_ms']
        or cfg['local_z'] <= 0 or cfg['local_rms_ratio'] <= 1 or cfg['local_peak_ratio'] <= 1
        or args.plots < 0):
        ap.error('窗口/阈值/声道参数不合法。')
    cfg['channel'] = int(cfg['channel'])
    if args.quiet_reference:
        try:
            _, reference_x, _, _, _ = read_signal(args.quiet_reference, cfg)
            reference_rms = float(np.sqrt(np.mean(reference_x*reference_x)))
            reference_peak = float(np.max(np.abs(reference_x)))
        except (ValueError, OSError) as exc:
            ap.error(f'停机参考音频读取失败：{exc}')
        if reference_peak <= cfg['silence_peak']:
            print('停机参考近乎全零，无法估计底噪，保留当前门控阈值。')
        else:
            cfg['quiet_rms'] = 3 * reference_rms
            cfg['quiet_peak'] = 3 * reference_peak
            print(f'停机参考校准：quiet_rms={cfg["quiet_rms"]:.6g}, quiet_peak={cfg["quiet_peak"]:.6g}')
    root, output = args.input.resolve(), args.output.resolve()
    if not root.is_dir() or root == output or root.is_relative_to(output):
        ap.error('输入须为目录；输出不能是输入目录本身或其祖先目录。')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        ap.error('输出目录必须为空或不存在；请换一个新目录，避免混入旧结果。')
    output.mkdir(parents=True, exist_ok=True)
    (output/'model.json').write_text(json.dumps({'version': VERSION, 'method': 'within_file_morphology',
                                                'config': cfg}, ensure_ascii=False, indent=2), encoding='utf-8')
    (output/'run_config.json').write_text(json.dumps({'input': str(root), 'rename_mode': args.rename_mode,
                       'quiet_reference': str(args.quiet_reference.resolve()) if args.quiet_reference else None,
                       'copy_classified': not args.no_copy, 'config': cfg}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'重命名模式：{args.rename_mode}；输出目录：{output}', flush=True)
    entries = prepare_files(root, output, args.rename_mode)
    results, errors, events = [], [], []
    frames_dir = output/'frames'
    frames_dir.mkdir()
    for i, entry in enumerate(entries):
        path = Path(entry['analysis_path'])
        try:
            row, frame = classify(path, cfg)
            row.update({k: entry[k] for k in ['file','original_file','frame_id','rename_status']})
        except Exception as exc:
            errors.append({'original_file': entry['original_file'], 'file': entry['file'],
                           'error': str(exc), 'status': '读取/分析失败，未强行归类'})
            print(f'[{i+1}/{len(entries)}] 失败：{entry["file"]} | {exc}', flush=True)
            continue
        # 特征提取异常与输出写入失败分开处理：磁盘写入失败应明确中止，不能冒充坏音频。
        frame.to_csv(frames_dir/f"{row['frame_id']}.csv", index=False, encoding='utf-8-sig')
        for start, end in merge_events(frame):
            events.append({'file': row['file'], 'start_s': start, 'end_s': end})
        if not args.no_copy:
            category = 'nonstationary' if row['label'] == '非平稳型' else 'stable'
            # 保留原子目录结构；极端改名失败引起路径碰撞时再加文件编号。
            target = output/'classified'/category/row['file']
            if target.exists():
                target = target.with_name(f"{target.stem}_{row['frame_id']}{target.suffix}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            row['classified_file'] = target.relative_to(output).as_posix()
        results.append(row)
        print(f'[{i+1}/{len(entries)}] {row["file"]} -> {row["label"]}', flush=True)
    result = pd.DataFrame(results)
    if result.empty:
        result = pd.DataFrame(columns=['original_file','file','frame_id','label','score','reason'])
    if len(result):
        result = result.sort_values('score', ascending=False)
    result.to_csv(output/'results.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(errors, columns=['original_file','file','error','status']).to_csv(output/'errors.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(events, columns=['file','start_s','end_s']).to_csv(output/'events.csv', index=False, encoding='utf-8-sig')
    # CSV 已落盘，再生成图；绘图失败单独记录，不影响已经完成的分类。
    plot_errors = []
    lookup = {e['frame_id']: Path(e['analysis_path']) for e in entries}
    if len(result) and args.plots:
        (output/'plots').mkdir()
        for row in result.head(args.plots).to_dict('records'):
            try:
                frame = pd.read_csv(frames_dir/f"{row['frame_id']}.csv")
                plot_file(lookup[row['frame_id']], row, frame, cfg, output/'plots'/f"{row['frame_id']}.png")
            except Exception as exc:
                plot_errors.append({'file': row['file'], 'error': str(exc)})
    pd.DataFrame(plot_errors, columns=['file','error']).to_csv(output/'plot_errors.csv', index=False, encoding='utf-8-sig')
    labels = {name: int((result.label == name).sum()) if len(result) else 0 for name in ['平稳型','非平稳型']}
    summary = {'total_files': len(entries), 'valid_files': len(results), 'error_files': len(errors),
               'labels': labels, 'silent_stable_files': int(result.is_silent.sum()) if len(result) else 0,
               'low_energy_stable_files': int(result.quiet_gate.sum()) if len(result) else 0,
               'rename_failures': sum(e['rename_status'] == 'failed' for e in entries),
               'plot_failures': len(plot_errors),
               'warning': '波形形态二分类；静音包含在平稳型中；不代表故障诊断或已验证准确率。'}
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
