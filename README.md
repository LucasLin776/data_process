# 螺栓掉落声音数据增强

用人工标注的螺栓掉落事件，与真实正常运行背景按指定信噪比混合，并加入低、中、高频段的轻微随机均衡；输出连续编号的WAV、时域波形图、STFT时频图、时间标签、训练清单和来源/质量日志。这个目录只准备训练数据，不训练检测模型。

## 1. 当前已确认的数据

`augmentation.json` 中的路径相对于该配置文件，而不是终端当前目录：

| 角色 | 原文件 | 用途 |
|---|---|---|
| 螺栓正例 | `../螺栓掉落.wav` | 目前只有一个真实正例录音 |
| 正例标注 | `../标记.csv` | 用户确认对应上述录音；12.808～17.572秒 |
| 正常背景 | `../数据增强/低正常.wav`、`中正常.wav`、`中正常2.wav`、`高正常.wav` | 与事件混合，也保留正常负例 |
| 干扰负例 | `../数据增强/胶粒掉落.wav`、`胶粒掉落2.wav`、`胶粒掉落3.wav`、`胶粒掉落4.wav` | 用户确认没有螺栓；作为困难负例，不当正常背景叠加 |

原文件只读使用；不改名、覆盖或删除第二个文件夹、桌面正例和标记文件。

## 2. 用 Conda 运行

在 Miniconda Prompt 中：

```bat
conda activate wind-bolt-audio
cd /d C:\Users\tlzn\Desktop\data_process
python -m pip install -r requirements.txt
python augment.py generate --config augmentation.json --output output\bolt_aug_v2_eq
```

`output\bolt_aug_v1` 是未加入EQ的旧批次；本次随机均衡批次使用`output\bolt_aug_v2_eq`。再次运行请换一个新目录。输出目录已存在时会报错，不覆盖旧结果。依赖仍为NumPy、SciPy、SoundFile和Matplotlib；均衡使用现有SciPy，不新增依赖，做增强不需要安装PyTorch或GPU。

只扫描音频质量：

```bat
python augment.py scan --input "C:\Users\tlzn\Desktop\数据增强" --output reports\inventory_v2.json
```

扫描不按文件名自动标注类别，报告存在时也不覆盖。

## 3. 默认增强流程

1. 读取第0声道、去直流、抗混叠重采样到32 kHz；不按每条峰值归一化，不做激进降噪。
2. 读取Audition标记，截取事件及前后最多20 ms上下文；5 ms淡入淡出只放在上下文中，不抹掉标注区间内的冲击。
3. 从正常背景取30秒。源长度不足时拒绝，不循环背景或补零凑长度。
4. 将事件放进完整30秒时间轴，抽取低、中、高频增益，对前景和背景应用同一个均衡器；在均衡后的标注事件区间计算前景和背景功率，设置-5、0、5、10 dB，每档随机放置2次。
5. 每条混合正例配一条不插入事件的负例，两条使用相同背景片段、相同EQ和相同全局增益，降低均衡效果或背景响度成为类别线索的风险。
6. 按正例和配对负例的共同峰值计算必要衰减，再叠加随机-6～0 dB全局增益；不硬裁剪、不压缩动态，不把每条音频都归一化到同一峰值。
7. 保留一份每条真实来源的统一格式版本，并对每条来源额外生成一个随机EQ版本，包括螺栓正例、正常背景和胶粒干扰负例。每写出一条WAV，立即读取实际写出的音频生成波形图和时频图，再记录来源、EQ参数和质量日志。

公式：`mixed = (EQ(background) + alpha * EQ(foreground_shifted)) * common_gain`。正负配对使用同一组EQ参数。

`alpha = sqrt(background_power / foreground_power) * 10 ** (snr_db / 20)`。功率在EQ后的事件区间计算，不用整条30秒平均，避免均衡后SNR悄悄偏离设定。源事件含残余背景和原始失真，所以这里的SNR是前景片段/背景的近似功率比，不是已分离干净声源的物理测量值。

原标记12.808～17.572秒用于截取来源事件；混合后的标签跟随随机位置变化，不继续使用原录音时间。边界对齐到32 kHz采样点。

当前默认数量：

- 1个来源事件 × 4条正常背景 × 4档SNR × 2次 = **32条混合正例**。
- **32条配对背景负例**。
- **9条来源格式统一版本**：1条正例、4条正常、4条干扰负例。
- **9条来源随机EQ版本**：1条正例、4条正常、4条干扰负例。
- 合计 **82条**，34条含目标、48条不含目标；其中73条应用了EQ、9条只做来源格式统一；全部30秒、32 kHz、单声道、PCM16。

82条不等于82次独立试验，真实正例来源仍只有1条。关闭`random_eq.enabled`或使用不含`random_eq`的旧配置时，恢复原先的73条流程。

### 随机均衡的频段与标签

默认三段增益分别独立均匀抽取自`[-3, 3] dB`。交界频率为500 Hz和4 kHz，过渡带为400～600 Hz、3.2～4.8 kHz；频段之间平滑过渡，不在频域直接硬切。该范围是轻量初始设置，仍需用独立真实录音验证。

使用SciPy `firwin2`设计513抽头、Hann窗线性相位FIR，用`oaconvolve`滤波，对称填充并补偿256采样点延迟。音频时长和事件时间轴不变，不变调、不伸缩。32 kHz下滤波可能在瞬态两侧各产生最多8 ms的扩展，因此没有宣称冲击边缘逐点不变；标签继续表示人工确认的原始事件区间。前景先放入完整时间轴再滤波，避免在裁剪边界截断响应。

每个配对共享EQ；每条来源额外抽取独立EQ。均衡采用独立随机流，因此开关EQ不会改变混合的背景裁剪、插入时间和随机全局增益。日志中`augmentation.random_eq`记录三段增益、交界频率、过渡带、滤波器长度和延迟补偿；来源统一格式版本记录`applied: false`。

## 4. 输出文件

```text
output/bolt_aug_v2_eq/
  audio/                   01.wav、02.wav……82.wav
  waveforms/               01.png、02.png……82.png，时域波形图
  spectrograms/            01.png、02.png……82.png，STFT时频图
  labels/                  34份正例对应的同编号Audition CSV
  train.jsonl              兼容 D:/project_sound 的数据清单格式
  augmentation_log.jsonl   每条的EQ、SNR、位置、增益、配对关系和质量
  quality.csv              输出音频质量表
  sources.json             来源路径、SHA256、原始标签及源质量
  config_used.json          配置快照
  summary.json              数量、质量提示、耗时和完成状态
  renaming_map.csv          仅旧批次编号迁移时存在，新批次没有此文件
```

文件名从`01`开始逐条编号，不包含类别、来源、信噪比或配对关系；超过99条后继续使用`100`、`101`等。比如`audio/01.wav`对应`waveforms/01.png`、`spectrograms/01.png`，如果含事件则另有`labels/01.csv`。

`recording_id`与编号一致。原来的`original_`、`pos_`、`neg_`等描述保存在`derivation_id`字段，来源与配对关系保存在`source_ids`和增强日志中。打乱训练读取顺序时，按清单保持音频、标签的对应关系。

两类图片都只显示编号，标题不写类别或信噪比。波形图横轴为秒、纵轴为数字满量程比例，固定范围约为-1～1；长音频按块画最小/最大包络，保留短冲击峰值。时频图使用Hann窗STFT，默认1024采样点窗口、256采样点步长（32 kHz下为32 ms/8 ms），覆盖0～16 kHz；色标固定-100～0 dB，相对于1 FS的STFT幅值，不是经过声学校准的dB SPL。图片只用于查看，不替代模型的特征提取。

只有正例有范围标记；负例由清单中的`events: []`明确表示，不伪造空标签。一条WAV一条JSONL记录；一个原文件的多个标记分别参与混合，来源正例版本保留全部标记。

`reviewed: true`表示来源类别和标注已确认、输出标签确定性转换，**不代表每条生成音频已人工试听**。`annotation.output_listened_by_human`为false。建议试听各背景/SNR下的代表样本再决定是否用于训练。

运行中断时目录保留`RUNNING.json`，不要用于训练；成功时移除此标志，`summary.json`的`status`必须为`complete`。

### 已有结果改名与补图

```bat
python augment.py renumber --input output\bolt_aug_v1
```

命令在临时副本中同步更新WAV名、标记名、清单、日志和质量表，补齐两类图片；核对WAV的SHA256并完整备份后，逐文件替换，兼容Windows中打开的目录。音频内容和事件起止不改变，旧结果备份到`output/_backups/`，具体路径写入`summary.json`的`renumber_backup`。迁移完成后再次执行会报告`unchanged`。本次73条已有结果已经执行完成，不需要你再操作。迁移中`RENUMBERING.json`表示尚未完成；文件替换失败时会尝试从备份回滚。

## 5. 接入已有检测项目

在`D:\project_sound`中、已激活`wind-bolt-audio`后检查：

```bat
python -m bolt_audio check-data --manifest "C:\Users\tlzn\Desktop\data_process\output\bolt_aug_v2_eq\train.jsonl"
```

之后按主项目提取特征、训练。**此清单只有train，不能单独满足当前训练器要求独立val数据的条件。** 要另补真实验证录音再组合清单；组合时将本清单的相对`audio`路径转为绝对路径，或保留原相对目录关系。仅复制清单到另一目录会导致路径失效。

当前所有来源和派生文件统一`split=train`、同一个`group_id`，防止增强版本跨划分。不要随机把本次WAV分成训练/验证/测试，也不要复制文件伪造验证集。以后按真实采集会话先划分来源，再增强训练部分。本工具拒绝显式标为val/test的来源。

## 6. 后续调整配置

修改`augmentation.json`：

- `snr_db`：实验信噪比档位，需要现场验证，不保证每档都可听辨。
- `repeats_per_snr`：每个事件/背景/SNR组合的随机次数；增加次数不增加真实事件多样性。
- `seed`：相同配置、原文件、软件版本和种子可复现WAV。
- `output_gain_db`：全局增益范围，只允许衰减。
- `random_eq.enabled`：是否开启随机均衡，缺省为false。
- `random_eq.gain_db`：三段各自独立抽样的增益范围，默认`[-3, 3]`dB；配置限制在`[-6, 6]`dB内。
- `random_eq.crossovers_hz`：低/中、中/高交界，默认`[500, 4000]`Hz；连同过渡带必须低于采样率一半。
- `random_eq.transition_fraction`：交界两侧的过渡比例，默认0.2。
- `random_eq.fir_taps`：FIR奇数抽头数，默认513；修改会改变过渡精度和瞬态扩展范围。
- `random_eq.source_variants`：每条来源额外生成的EQ版本数，默认1；混合配对每次也会抽取EQ。
- `foregrounds`：增加真实螺栓录音和对应`markers`；也可提供`events: [{"start_s": ..., "end_s": ...}]`，二者只能选一个。
- `backgrounds`：仅放确认无目标且适合作混合背景的录音。
- `hard_negatives`：无螺栓的相似干扰录音，保留负例，不进行目标混合。

来源ID唯一，重复音频按SHA256拒绝。Audition支持UTF-8 BOM、Tab或逗号分列；Name=`bolt_drop`、Time Format=`decimal`、Duration>0。空标记、点标记、重叠和明显越界均报错。暂不加入变调、时间伸缩或频谱遮挡，先保留真实冲击结构。

部分来源存在近满量程样本，详见`sources.json`。避免新增削波不能修复原始失真，静止事件叠加背景也不能完整还原运行中的结构传声。

## 7. 代码阅读与检查

- `augment.py`：`read_audio`读取/质量检查，`read_markers`标注解析，`event_clip`截取，`random_eq_settings`校验配置，`sample_equalizer`抽样和设计EQ，`apply_equalizer`滤波与延迟补偿，`mix_at_snr`均衡后混合，`render_audio_plots`生成两类图片，`generate`整套输出，`renumber_existing`迁移既有结果。
- `tests/test_augment.py`：EQ频段响应、冲击对齐、无循环回绕、EQ后SNR、配对共享参数、各类别覆盖、标签不偏移、配置校验，以及原有来源隔离、复现性、输出质量和编号迁移检查。

```bat
python -m unittest discover -s tests -v
```

这些检查验证数据处理逻辑，不证明检测准确率。

## 8. 本次清理记录

以下内容移到`C:\Users\tlzn\Desktop\data_process_backup_20261026_augmentation`，从项目移出，未永久删除：

- `1276996795343581185/`：旧批次原始数据，1124个文件，约2.97 GB。
- `datas/`：旧输入数据，154个文件，约0.41 GB。
- `output/`：旧筛查输出，3390个文件，约6.19 GB。
- `plot_sum/`：旧图表和信息，4个文件。
- `detect.py`：原波形平稳/非平稳筛查脚本，不能直接充当螺栓检测器。
- `plot.py`：原单音频绘图脚本。
- `validate.py`：旧算法检查，与本次增强无关。

旧README、requirements、.gitignore也已备份。保留原`.git`；没有stage、commit或push。
