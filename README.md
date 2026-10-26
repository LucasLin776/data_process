# 螺栓掉落声音数据增强

用人工标注的螺栓掉落事件，与真实正常运行背景按指定信噪比混合；输出WAV、时间标签、训练清单和来源/质量日志。这个目录只准备训练数据，不训练检测模型。

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
python augment.py generate --config augmentation.json --output output\bolt_aug_v2
```

`output\bolt_aug_v1` 已由本次实际运行生成，重新运行请换一个新目录。输出目录已存在时会报错，不覆盖旧结果。依赖只有NumPy、SciPy、SoundFile；做增强不需要安装PyTorch或GPU。

只扫描音频质量：

```bat
python augment.py scan --input "C:\Users\tlzn\Desktop\数据增强" --output reports\inventory_v2.json
```

扫描不按文件名自动标注类别，报告存在时也不覆盖。

## 3. 默认增强流程

1. 读取第0声道、去直流、抗混叠重采样到32 kHz；不按每条峰值归一化，不做激进降噪。
2. 读取Audition标记，截取事件及前后最多20 ms上下文；5 ms淡入淡出只放在上下文中，不抹掉标注区间内的冲击。
3. 从正常背景取30秒。源长度不足时拒绝，不循环背景或补零凑长度。
4. 按标注事件区间计算前景和对应背景功率，设置-5、0、5、10 dB，每档随机放置2次。
5. 每条混合正例配一条不插入事件的负例，两条使用相同背景片段、相同全局增益，降低背景响度成为类别线索的风险。
6. 按正例和配对负例的共同峰值计算必要衰减，再叠加随机-6～0 dB全局增益；不硬裁剪、不压缩动态，不把每条音频都归一化到同一峰值。
7. 保留一份每条真实来源的统一格式版本，并生成来源、变换参数和质量日志。

公式：`mixed = (background + alpha * foreground_shifted) * common_gain`。

`alpha = sqrt(background_power / foreground_power) * 10 ** (snr_db / 20)`。功率只在事件区间计算，不用整条30秒平均。源事件含残余背景和原始失真，所以这里的SNR是前景片段/背景的近似功率比，不是已分离干净声源的物理测量值。

原标记12.808～17.572秒用于截取来源事件；混合后的标签跟随随机位置变化，不继续使用原录音时间。边界对齐到32 kHz采样点。

当前默认数量：

- 1个来源事件 × 4条正常背景 × 4档SNR × 2次 = **32条混合正例**。
- **32条配对背景负例**。
- **9条来源格式统一版本**：1条正例、4条正常、4条干扰负例。
- 合计 **73条**，33条含目标、40条不含目标；全部30秒、32 kHz、单声道、PCM16。

73条不等于73次独立试验，真实正例来源仍只有1条。

## 4. 输出文件

```text
output/bolt_aug_v1/
  audio/                   73条WAV
  labels/                  33份含事件WAV对应的Audition CSV
  train.jsonl              兼容 D:/project_sound 的数据清单格式
  augmentation_log.jsonl   每条的SNR、位置、增益、配对关系和质量
  quality.csv              输出音频质量表
  sources.json             来源路径、SHA256、原始标签及源质量
  config_used.json          配置快照
  summary.json              数量、质量提示、耗时和完成状态
```

`original_`是来源格式统一版本，`pos_`是混合正例，`neg_`是配对背景负例。干扰负例前缀为`original_interference_`。

只有正例有范围标记；负例由清单中的`events: []`明确表示，不伪造空标签。一条WAV一条JSONL记录；一个原文件的多个标记分别参与混合，来源正例版本保留全部标记。

`reviewed: true`表示来源类别和标注已确认、输出标签确定性转换，**不代表每条生成音频已人工试听**。`annotation.output_listened_by_human`为false。建议试听各背景/SNR下的代表样本再决定是否用于训练。

运行中断时目录保留`RUNNING.json`，不要用于训练；成功时移除此标志，`summary.json`的`status`必须为`complete`。

## 5. 接入已有检测项目

在`D:\project_sound`中、已激活`wind-bolt-audio`后检查：

```bat
python -m bolt_audio check-data --manifest "C:\Users\tlzn\Desktop\data_process\output\bolt_aug_v1\train.jsonl"
```

之后按主项目提取特征、训练。**此清单只有train，不能单独满足当前训练器要求独立val数据的条件。** 要另补真实验证录音再组合清单；组合时将本清单的相对`audio`路径转为绝对路径，或保留原相对目录关系。仅复制清单到另一目录会导致路径失效。

当前所有来源和派生文件统一`split=train`、同一个`group_id`，防止增强版本跨划分。不要随机把本次WAV分成训练/验证/测试，也不要复制文件伪造验证集。以后按真实采集会话先划分来源，再增强训练部分。本工具拒绝显式标为val/test的来源。

## 6. 后续调整配置

修改`augmentation.json`：

- `snr_db`：实验信噪比档位，需要现场验证，不保证每档都可听辨。
- `repeats_per_snr`：每个事件/背景/SNR组合的随机次数；增加次数不增加真实事件多样性。
- `seed`：相同配置、原文件、软件版本和种子可复现WAV。
- `output_gain_db`：全局增益范围，只允许衰减。
- `foregrounds`：增加真实螺栓录音和对应`markers`；也可提供`events: [{"start_s": ..., "end_s": ...}]`，二者只能选一个。
- `backgrounds`：仅放确认无目标且适合作混合背景的录音。
- `hard_negatives`：无螺栓的相似干扰录音，保留负例，不进行目标混合。

来源ID唯一，重复音频按SHA256拒绝。Audition支持UTF-8 BOM、Tab或逗号分列；Name=`bolt_drop`、Time Format=`decimal`、Duration>0。空标记、点标记、重叠和明显越界均报错。暂不加入变调、时间伸缩或频谱遮挡，先保留真实冲击结构。

部分来源存在近满量程样本，详见`sources.json`。避免新增削波不能修复原始失真，静止事件叠加背景也不能完整还原运行中的结构传声。

## 7. 代码阅读与检查

- `augment.py`：`read_audio`读取/质量检查，`read_markers`标注解析，`event_clip`截取，`mix_at_snr`混合，`generate`整套输出。
- `tests/test_augment.py`：SNR、配对增益、标签往返、多事件、拒绝条件、重采样、复现性和输出质量检查。

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
