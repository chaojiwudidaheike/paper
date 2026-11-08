# DeepSeek Skill MDC 调度

基于 `CpOL-AsHs-Coarse-BGUP-Cache`；Tools.py、algorithm.py、coarse_scheduler.py
为基准源文件的逐字复制。新 runner 复用缓存建图、BGUP、节点权重、MWIS、更新逻辑。
默认 Random_1、scope=20、TimeLimit=120、SingleStep=10；当前实验关闭全部故障。
没有 MDC 速度限制，没有滚动时域模拟，保留移动距离软惩罚。

## 1. 首次配置运行环境

以下命令均在项目的 `MDC` 目录执行。本机路径如下；他人复现时替换成自己的下载路径：

```bash
cd /Users/yfjiazi/res-project/LLM-MDC/MDC
```

项目用 uv 管理 Python 和依赖，`pyproject.toml` 要求 Python >= 3.12，
`.python-version` 指定 3.12。依赖包括 NumPy、SciPy、NetworkX 和 PyTorch。
本算法通过 Python 标准库调用 DeepSeek REST 接口，无需额外安装 API SDK。

如果还没安装 uv，可在已有 Python 3 的环境中执行：

```bash
python3 -m pip install uv
```

确认 uv 可用，然后安装项目指定的 Python 和锁定版本的依赖：

```bash
uv --version
uv python install 3.12
uv sync --locked
```

首次安装需要联网，会在 `MDC/.venv` 创建虚拟环境。后续使用 `uv run` 无需手动激活。
如果找不到 `uv` 命令，可将本文命令中的 `uv` 换成 `python3 -m uv`。
已有环境且依赖安装完整时，无需重复安装；下面启动命令的 `--no-sync` 表示跳过依赖同步。
如果缺少依赖，应先重新执行 `uv sync --locked`。

## 2. 在项目中保存 API Key（只需填写一次）

密钥文件位于 `MDC/.env`，本机完整路径：

```text
/Users/yfjiazi/res-project/LLM-MDC/MDC/.env
```

在 `MDC` 目录执行：

```bash
nano .env
```

如果下载的项目没有 `.env`，上述命令会创建它。在文件中填写下面这一行，
将等号右侧的占位文字换成真实密钥；不要写 `export`：

```dotenv
DEEPSEEK_API_KEY=这里填写你的真实密钥
```

按 `Control + O`、回车保存，再按 `Control + X` 退出。随后执行：

```bash
chmod 600 .env
```

以后重新打开终端也无需重新填写密钥。程序本身不自动读取 `.env`，
下文启动命令通过 `uv run --env-file .env` 加载它。
若当前终端之前通过 `export` 设置过密钥，先执行 `unset DEEPSEEK_API_KEY`，
避免终端中的旧值覆盖文件配置；该命令不会删除文件里的密钥。

`MDC/.gitignore` 已忽略 `.env`。手动打包、上传或分享文件前，仍需清空密钥值，
只保留如下内容供其他人填写，不要把真实密钥写进源码、README、skill 或结果文件：

```dotenv
DEEPSEEK_API_KEY=
```

## 3. 启动在线实验

完成环境安装和密钥填写后，在 `MDC` 目录复制执行：

```bash
unset DEEPSEEK_API_KEY
uv run --env-file .env --no-sync python code/CpOL-AsHs-DeepSeek-Skill/runner.py \
  --dataset data/Mat/Random_1.mat \
  --time-limit 120 \
  --interval 10 \
  --scope 20 \
  --episodes 3 \
  --state code/CpOL-AsHs-DeepSeek-Skill/state/random1_t120_i10_s20
```

此命令完整运行 3 轮，每轮 12 个调度步，每轮结束后评估公式和权重更新。
未触发重试时约 39 次 API 请求（每轮 12 次位置决策 + 1 次策略提议；回放评估不调用模型），
会产生 API 费用。仿真时长不等于实际运行耗时，实际耗时包含计算和 API 请求等待。

仅用默认参数运行 1 轮的简短命令：

```bash
uv run --env-file .env --no-sync python code/CpOL-AsHs-DeepSeek-Skill/runner.py
```

## 4. 更换数据集和实验参数

| 参数 | 含义 | 程序默认值 |
|---|---|---|
| `--dataset` | `.mat` 数据集路径；显式传入的相对路径以当前工作目录为基准 | `MDC/data/Mat/Random_1.mat` |
| `--time-limit` | 仿真总时长，单位沿用数据集时间尺度 | `120` |
| `--interval` | 每次调度之间的仿真时间间隔 | `10` |
| `--scope` | MDC 正方形覆盖区域边长；20 表示 20 × 20 网格，不是半径 | `20` |
| `--episodes` | 完整仿真轮数，每轮结束评估更新 | `1` |
| `--state` | 公式、权重和日志的保存及读取目录 | 本算法目录下的 `state/` |
| `--model` | DeepSeek 模型名称 | `DEEPSEEK_MODEL` 环境变量；未设置时为 `deepseek-flash` |
| `--select-max-tokens` | 位置选择请求的初始生成 token 上限；截断时翻倍重试一次 | `4096` |
| `--evolve-max-tokens` | 策略更新请求的初始生成 token 上限；截断时翻倍重试一次 | `16384` |

时长、间隔和轮数必须是正整数，时长必须能被间隔整除。
覆盖边长必须在数据集网格尺寸允许的范围内。
数据集位于 `MDC/data/Mat/`，注意文件名大小写，例如 `Random_1.mat`。

例如，换成 Random_2，时长 180、间隔 5、覆盖 30 × 30、运行 3 轮：

```bash
uv run --env-file .env --no-sync python code/CpOL-AsHs-DeepSeek-Skill/runner.py \
  --dataset data/Mat/Random_2.mat \
  --time-limit 180 \
  --interval 5 \
  --scope 30 \
  --episodes 3 \
  --state code/CpOL-AsHs-DeepSeek-Skill/state/random2_t180_i5_s30
```

同一个 `--state` 目录会继续使用之前保存的策略和反馈。不同数据集或参数组合建议使用不同目录；
若要从初始策略重新运行，请指定一个尚未使用的新目录。为了复现实验，请保留启动参数、
数据集、`uv.lock`、实际模型名称和状态日志；在线模型提议可能变化，不保证逐次结果完全一致。

## 5. 离线检查与常见问题

以下命令也在 `MDC` 目录执行，不需要 API Key。离线模式仅运行本地评分调度，不调用模型或生成策略更新：

```bash
uv run --no-sync python code/CpOL-AsHs-DeepSeek-Skill/runner.py --offline \
  --state code/CpOL-AsHs-DeepSeek-Skill/state/offline_check
uv run --no-sync python -m unittest discover -s code/CpOL-AsHs-DeepSeek-Skill -p 'test_*.py'
```

- 提示未设置 `DEEPSEEK_API_KEY`：检查 `.env` 中等号右侧是否已填写，并确认启动命令包含 `--env-file .env`。
- 找不到 `.env`、数据集或脚本：确认当前位于 `MDC` 目录。
- 提示模块缺失：在 `MDC` 目录执行 `uv sync --locked` 后重新启动。
- API 认证或请求失败：检查密钥、账户可用额度、模型权限和网络连接。程序会报错，不会自动改为离线运行。

### 响应被截断：`DeepSeek response incomplete: length`

这表示 API 响应触及长度限制，不代表新策略的任务完成数下降。此前代码对所有请求
固定使用 `max_tokens=4096`；现在位置选择和策略更新分别配置预算，策略更新默认使用 16384。
收到 `finish_reason=length` 时，将本次请求预算翻倍，并从头重试一次；不会拼接或采用截断内容。
若再次截断，仍报错；发生在轮末策略更新时保留旧策略和本轮正式仿真结果。
其他结束原因不会触发此次长度重试。

原启动命令无需修改即可使用新默认值。如果仍遇到截断，可尝试：

```bash
uv run --env-file .env --no-sync python code/CpOL-AsHs-DeepSeek-Skill/runner.py \
  --time-limit 180 \
  --episodes 3 \
  --evolve-max-tokens 32768 \
  --state code/CpOL-AsHs-DeepSeek-Skill/state
```

该设置会在首次截断后以 65536 上限重试。参数需为正整数，初始及翻倍后的上限均应在
所选模型允许的范围内。重试会增加 API 请求、费用和等待时间；增大预算无法保证解决上下文限制等所有截断原因。
Skill 同时要求简短理由，不回显输入历史。完整响应仍须通过 JSON、公式和回放接纳检查。

终端会显示策略更新请求的 `finish_reason`、`max_tokens` 和 API 返回的 `usage`。
每轮 JSON 的 `evolution_api_attempts` 保存这些诊断信息，不保存密钥或截断正文，
便于区分请求失败与策略效果未提升。正在运行的旧进程不会加载修改，需要重新启动。

代码使用的接口为 `https://api.deepseek.com/chat/completions`。
服务说明见 [DeepSeek 官方文档](https://api-docs.deepseek.com/) 和
[JSON 输出文档](https://api-docs.deepseek.com/guides/json_mode/)。

## Skill 如何生效

`skills/mdc-scheduling/SKILL.md` 是算法自带的运行时 skill，不要求安装进 Codex。
每次 API 请求均完整加载它到 system message；当前策略、候选区域、近期反馈放在 user message。
DeepSeek 返回 JSON 位置和理由；宿主检查该位置确为最高评分候选（同分使用固定顺序）。
数值计算由本地执行，模型负责遵循 skill 输出位置和基于反馈提出策略更新。
响应为空、截断、位置无效或 API 失败会显式报错，不静默替换为本地结果。

## 公式和权重迭代

每轮固定策略执行完毕后，模型提出完整六项公式和六个权重。
公式可使用原始特征及任务数量、可行组合数量、时间余量、平均耗时/能耗等变量，
支持算术、min/max、abs、sqrt、log1p、tanh。它能改变六个评分函数的表达式结构，
不是仅选择固定公式；底层可行性和物理约束不由模型修改。具体语法见 skill。

新旧策略在同一初始数据、时间、覆盖范围上各完整回放一次；任务完成数严格增加才接纳。
评估用与模型选择相同的确定性最高分规则，省去回放 API 开销。每次调用仍记录来源，
真实部署轮次使用 DeepSeek 输出。等分和退化提议均拒绝；非法公式保留错误反馈。
下一轮/下次运行加载已接纳版本，也把最近五次提议结果提供给模型。

默认 `state/` 保存：

- `policy.json`：最新接纳的公式和权重，原子替换。
- `last_feedback.json`：最近五轮提议和验证反馈（包括拒绝原因）。
- `<session>-<episode>.json`：本轮旧策略、模型理由、每步位置/分项分数/任务数、提议、新旧回放。

可用 `--state PATH` 为不同数据或实验指定独立学习状态。更新不保证每轮成功，
也不保证迁移到新数据有效；本实现是同一适应数据上的自动搜索，正式报告应另设未参与更新的测试集。
ATRT 的历史口径问题未改，接纳指标仅使用任务完成数。
仿真继续继承基准的时间检查和任务计数语义，不把本实验解释为修复全部仿真问题。
