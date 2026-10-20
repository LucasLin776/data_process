"""
快速查看单个 WAV 的时域波形与频域分布（单边 FFT 幅度谱）。

依赖安装：python -m pip install numpy scipy matplotlib
基本运行：python plot_wav.py --input "D:/hub_wav/example.wav"
指定输出：python plot_wav.py --input "D:/hub_wav/example.wav" --output "D:/wav_plot"
截取分析：python plot_wav.py --input "D:/hub_wav/example.wav" --start 2 --duration 5

默认分析第 0 声道、去直流、使用 Hann 窗计算频谱，保存三张 PNG 和信息 JSON。
图内使用英文，避免部分电脑缺少中文字体；函数和参数说明使用中文。
幅值是数字满量程比例，未经传感器声学校准，不是声压 Pa 或声压级 dB SPL。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.io import wavfile
from scipy.signal.windows import hann


def load_segment(path, channel=0, start=0.0, duration=None, remove_dc=True):
    """读取指定声道/时间段，返回采样率、浮点信号、实际起点和原文件信息。

    优先使用内存映射读取，减少读取长 PCM 文件时的内存压力。
    24 位 PCM 等格式不支持内存映射时，回退到普通读取。
    """
    with warnings.catch_warnings():
        warnings.simplefilter('error', wavfile.WavFileWarning)
        try:
            sr, raw = wavfile.read(path, mmap=True)
        except ValueError:
            sr, raw = wavfile.read(path, mmap=False)

    if raw.ndim not in (1, 2) or len(raw) == 0:
        raise ValueError('WAV 没有有效音频样本。')
    channels = 1 if raw.ndim == 1 else raw.shape[1]
    if not 0 <= channel < channels:
        raise ValueError(f'该音频只有 {channels} 个声道，声道编号从 0 开始。')

    # 起止点对齐到采样点。超出文件结尾的 duration 自动截到文件末尾。
    first = round(start * sr)
    stop = len(raw) if duration is None else min(len(raw), first + round(duration * sr))
    if first >= len(raw) or stop - first < 2:
        raise ValueError('选择的时间段不足 2 个样本，或起点超出了音频长度。')
    part = raw[first:stop] if raw.ndim == 1 else raw[first:stop, channel]

    # 按数据类型满量程转换，不除以本条音频的最大值，保留录音幅值关系。
    # WAV 的 uint8 零点是 128；有符号整数的零点是 0。
    if part.dtype.kind == 'u':
        half = (np.iinfo(part.dtype).max + 1) / 2
        x = (part.astype(np.float64) - half) / half
    elif part.dtype.kind == 'i':
        limits = np.iinfo(part.dtype)
        x = part.astype(np.float64) / max(abs(limits.min), limits.max)
    elif part.dtype.kind == 'f':
        x = part.astype(np.float64)
    else:
        raise ValueError(f'不支持的数据类型：{part.dtype}')
    if not np.isfinite(x).all():
        raise ValueError('音频包含 NaN 或无穷值，无法可靠绘图。')

    original_mean = float(x.mean())
    if remove_dc:
        x -= original_mean
    info = {
        'source': str(path.resolve()),
        'sample_rate_hz': int(sr),
        'channels': channels,
        'selected_channel': channel,
        'file_duration_s': len(raw) / sr,
        'segment_start_s': first / sr,
        'segment_duration_s': len(x) / sr,
        'segment_samples': len(x),
        'dc_removed': remove_dc,
        'original_segment_mean': original_mean,
    }
    return sr, x, first / sr, info


def amplitude_spectrum(x, sr):
    """计算 Hann 加窗、相干增益校正后的单边幅度谱。

    对 N 点信号计算 rFFT，频率间隔为 sr/N；不通过补零伪造分辨率。
    双边幅值先除以 sum(window)，再将有负频率镜像的正频率乘以 2。
    DC 不翻倍；偶数 N 的 Nyquist 点也不翻倍。

    对落在 FFT 频点上的稳态正弦，谱峰近似其峰值幅度。
    非整周期信号仍存在谱泄漏/栅栏效应；这里不是功率谱密度 PSD。
    """
    n = len(x)
    window = hann(n, sym=False)
    amplitude = np.abs(rfft(x * window)) / window.sum()
    if n % 2 == 0:
        amplitude[1:-1] *= 2
    else:
        amplitude[1:] *= 2
    return rfftfreq(n, d=1 / sr), amplitude


def draw_waveform(ax, x, sr, offset):
    """绘制时间波形；长音频采用分块最小/最大值包络，保留窄冲击。

    只压缩绘图数据，频谱仍使用选定时间段的全部样本。
    横轴为原始文件中的时间，而不是截取后的相对时间。
    """
    if len(x) <= 50000:
        ax.plot(offset + np.arange(len(x)) / sr, x, lw=.65, color='#2563eb')
    else:
        block = int(np.ceil(len(x) / 25000))
        starts = np.arange(0, len(x), block)
        lower = np.minimum.reduceat(x, starts)
        upper = np.maximum.reduceat(x, starts)
        ax.fill_between(offset + starts / sr, lower, upper, color='#2563eb')
    ax.set_title('Time-domain waveform')
    ax.set_xlabel('Time in original file (s)')
    ax.set_ylabel('Digital amplitude (full-scale units)')
    ax.set_xlim(offset, offset + len(x) / sr)
    ax.grid(alpha=.25)


def draw_spectrum(ax, freq, amplitude, min_freq, max_freq, linear):
    """绘制指定频率范围的单边幅度谱；默认使用数字幅值的分贝表示。

    dBFS = 20*log10(amplitude / 1)，参考幅度为数字峰值 1。
    仅将显示下限截为 -160 dBFS，避免 log10(0)，不修改原始谱数据。
    """
    mask = (freq >= min_freq) & (freq <= max_freq)
    values = amplitude if linear else 20 * np.log10(np.maximum(amplitude, 1e-8))
    ax.plot(freq[mask], values[mask], lw=.7, color='#e07820')
    ax.set_title('Single-sided amplitude spectrum (Hann window)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Peak amplitude (full-scale units)' if linear else 'Amplitude (dBFS, ref. peak = 1)')
    ax.set_xlim(min_freq, max_freq)
    ax.grid(alpha=.25)


def main():
    """读取参数、计算频谱、生成图片，并报告基本音频信息。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True, help='单个 WAV 文件路径')
    parser.add_argument('--output', type=Path, help='图片输出目录，默认 wav_plots/音频文件名')
    parser.add_argument('--channel', type=int, default=0, help='声道编号，从 0 开始')
    parser.add_argument('--start', type=float, default=0, help='分析起点，单位秒，默认 0')
    parser.add_argument('--duration', type=float, help='分析时长，单位秒，默认一直分析到末尾')
    parser.add_argument('--min-freq', type=float, default=0, help='频谱显示下限，单位 Hz')
    parser.add_argument('--max-freq', type=float, help='频谱显示上限，默认采样率的一半')
    parser.add_argument('--linear', action='store_true', help='频谱纵轴使用线性幅值，默认 dBFS')
    parser.add_argument('--keep-dc', action='store_true', help='保留直流，默认对分析段减去均值')
    parser.add_argument('--show', action='store_true', help='保存后弹出图片窗口，需要桌面绘图环境')
    args = parser.parse_args()

    numeric = [args.start, args.min_freq]
    numeric += [v for v in [args.duration, args.max_freq] if v is not None]
    if (not all(np.isfinite(numeric)) or args.start < 0 or args.channel < 0
            or args.min_freq < 0 or (args.duration is not None and args.duration <= 0)):
        parser.error('时间、频率和声道参数不合法，请使用有限的非负数及正的时长。')
    if not args.input.is_file():
        parser.error(f'找不到音频文件：{args.input}')
    try:
        sr, x, offset, info = load_segment(args.input, args.channel, args.start,
                                           args.duration, not args.keep_dc)
    except (ValueError, OSError, wavfile.WavFileWarning) as exc:
        parser.error(str(exc))
    high = sr / 2 if args.max_freq is None else args.max_freq
    if not 0 <= args.min_freq < high <= sr / 2:
        parser.error(f'需要 0 <= min-freq < max-freq <= {sr / 2:g} Hz。')

    freq, amp = amplitude_spectrum(x, sr)
    # 主频：在显示范围内寻找最强非直流频点，不推断它是机械故障频率。
    mask = (freq > 0) & (freq >= args.min_freq) & (freq <= high)
    candidates = np.flatnonzero(mask)
    peak = candidates[np.argmax(amp[candidates])] if len(candidates) else None
    has_peak = peak is not None and amp[peak] > 1e-15
    info.update({
        'rms': float(np.sqrt(np.mean(x * x))),
        'absolute_peak': float(np.max(np.abs(x))),
        'fft_bin_spacing_hz': sr / len(x),
        'window': 'periodic Hann, coherent-gain corrected',
        'display_frequency_range_hz': [args.min_freq, high],
        'dominant_non_dc_frequency_hz': float(freq[peak]) if has_peak else None,
        'dominant_peak_amplitude': float(amp[peak]) if has_peak else None,
        'spectrum_unit': 'digital peak amplitude' if args.linear else 'dBFS (peak reference 1)',
    })

    # 默认不弹窗，便于终端直接运行、服务器处理或脚本调用。
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = args.output or Path('wav_plots') / args.input.stem
    output.mkdir(parents=True, exist_ok=True)
    # 总览图：上方时域、下方频域，便于一次对照。
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(f'WAV analysis | channel {args.channel} | fs={sr} Hz | T={len(x)/sr:.3f} s')
    draw_waveform(axes[0], x, sr, offset)
    draw_spectrum(axes[1], freq, amp, args.min_freq, high, args.linear)
    fig.savefig(output / 'overview.png', dpi=160)

    # 分别保存两个独立图片，便于插入报告或放大检查。
    wave_fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    draw_waveform(ax, x, sr, offset)
    wave_fig.savefig(output / 'waveform.png', dpi=160)
    spec_fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    draw_spectrum(ax, freq, amp, args.min_freq, high, args.linear)
    spec_fig.savefig(output / 'spectrum.png', dpi=160)
    (output / 'info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'采样率：{sr} Hz；总声道：{info["channels"]}；分析声道：{args.channel}')
    print(f'原始时长：{info["file_duration_s"]:.3f} 秒；分析时长：{len(x)/sr:.3f} 秒')
    print(f'FFT 频点间隔：{sr/len(x):.6g} Hz（Hann 窗的实际分辨能力还受主瓣宽度影响）')
    print(f'显示范围内最强非直流频点：{freq[peak]:.3f} Hz' if has_peak else '未找到有效非直流谱峰。')
    print(f'已保存：{output.resolve()}')
    print('overview.png：总览；waveform.png：时域；spectrum.png：频域；info.json：参数信息')
    if args.show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
