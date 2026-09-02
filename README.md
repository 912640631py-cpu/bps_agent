# BPS 性能测试评估 Agent

基于 LangGraph 的 Keysight BreakingPoint 实机测试工具。它负责 BPS 打流、可选的 DUT 监控、报告导出、性能波动分析，并由 DeepSeek 给出最终 Verdict。

## 主要能力

- 从每个 Run 的 TOC 按“章节标题 + 父级路径”动态解析 Section ID。
- 分别导出主报告和秒级吞吐、Flow Rate、Concurrent Flows 时序数据。
- 在 Stable 阶段开始后的固定早期窗口建立并冻结 Tx/Rx 基线，进行确定性性能分析。
- LLM 返回 `retry` 时自动降载，最多执行六次；返回 `pass` 后立即结束。
- 使用 SQLite checkpoint 支持中断恢复和 Evidence 离线回放。
- 支持 BPS-only 模式，完全跳过 DUT 登录、CAPTCHA、keepalive 和监控读取。

## 安装

项目使用 Python 3.11+，本仓库约定使用 `shixi` Conda 环境：

```powershell
conda activate shixi
Set-Location E:\f_bps
python -m pip install -e '.[dev]'
```

## 配置与凭据

默认配置为 `config/demo.yaml`。其中 `bps.total_bandwidth_mbps` 表示首次打流目标，默认 400 Mbps。

模板的 100% 带宽不写死：程序通过 `getSharedComponentSettings` 读取 `totalBandwidth.originalValue`，再换算传给 `setSharedComponentSettings` 的百分比。目标不能超过模板原始带宽。

凭据保存在 Windows Credential Manager，不写入配置或审计文件：

- `BPS_USERNAME`、`BPS_PASSWORD`
- `DUT_BACKEND_USERNAME`、`DUT_BACKEND_PASSWORD`（默认 SSH 后端采集）
- `DUT_FRONTEND_USERNAME`、`DUT_FRONTEND_PASSWORD`（仅前端采集）
- `COMPANY_DEEPSEEK_API_KEY` 或 `DEEPSEEK_API_KEY`

`run` 会根据 BPS-only、DUT Collection Method 和 `--stop-before-llm` 只读取本次所需凭据。
缺失值会在启动时交互输入；全部输入完成后只询问一次是否将这些新值保存到
Windows Credential Manager。选择不保存时，新值仅用于本次运行；
`--stop-before-llm` 不要求 LLM API key。

```powershell
python -m bps_agent credentials set
python -m bps_agent credentials status
python -m bps_agent credentials delete
```

## 运行

```powershell
# 使用默认配置
python -m bps_agent run

# 临时覆盖模板、端口和首次流量目标
python -m bps_agent run -t TEMPLATE -p 4 5 -b 300

# 临时覆盖 DUT SSH 后端目标、接口和采样间隔
python -m bps_agent run -m backend_ssh `
  -dh 10.66.246.133 -dp 50023 `
  -i T1/1 -i T1/2 -s 10

# 使用 DUT 前端 API 采集
python -m bps_agent run -m frontend_api

# 完成实机测试和 Evidence，在调用 LLM 前停止
python -m bps_agent run -sb

# 在后台下载完整 BPS PDF 报告
python -m bps_agent run -f

# 只使用 BPS，不登录或读取 DUT
python -m bps_agent run -bo

# 恢复已有 Evaluation
python -m bps_agent run -r EVALUATION_ID

# 使用已有 Evidence 重新裁决
python -m bps_agent replay -e artifacts\EVALUATION_ID\attempt-01\evidence.json
```

`run` 参数：

| Short | Long | Description |
| --- | --- | --- |
| `-c FILE` | `--config FILE` | 配置文件 |
| `-r ID` | `--resume ID` | 恢复已有 Evaluation |
| `-t NAME` | `--template NAME` | BPS Template |
| `-p PORT...` | `--ports PORT...` | BPS 测试端口 |
| `-b MBPS` | `--total-bandwidth-mbps MBPS` | 首次目标带宽 |
| `-bo` | `--bps-only` | 跳过 DUT |
| `-m METHOD` | `--dut-collection-method METHOD` | DUT 采集方式 |
| `-dh HOST` | `--dut-host HOST` | DUT host |
| `-dp PORT` | `--dut-port PORT` | DUT SSH port |
| `-i IFACE` | `--dut-interface IFACE` | DUT interface，可重复 |
| `-s SEC` | `--dut-interval-seconds SEC` | DUT 采样周期 |
| `-sb` | `--stop-before-llm` | Evidence-only 模式 |
| `-f` | `--full-pdf` | 在后台下载完整 BPS PDF 报告 |

`replay` 参数：

| Short | Long | Description |
| --- | --- | --- |
| `-c FILE` | `--config FILE` | 配置文件 |
| `-e FILE` | `--evidence FILE` | Evidence JSON |

全局参数放在子命令之前，例如 `python -m bps_agent -v run`：

| Short | Long | Description |
| --- | --- | --- |
| `-v` | `--verbose` | Debug/详细日志 |
| `-h` | `--help` | 显示帮助 |

恢复运行时不能使用模板、端口、带宽、DUT 或模式覆盖参数。
也可以在 YAML 中设置 `evaluation.mode: bps_only`；默认值为 `bps_and_dut`。
`dut.collection_method` 默认是 `backend_ssh`，该模式必须显式配置 `dut.backend.host`。
当前 SSH 方式不读取、校验或持久化主机密钥。
设置为 `frontend_api` 时，CAPTCHA 由操作者输入且不会保存。

`llm.reasoning_effort` 控制模型推理程度，默认为 `max`，可在 `demo.yaml` 中调整。
端口互斥由 BPS 的非强制预留负责；项目不再维护本地端口锁文件。

## 降载策略

首次 Attempt 使用配置目标。若 LLM 判定未通过，后续目标依次为初始目标的 `80%`、`60%`、`40%`、`20%`、`10%`。

例如模板原始带宽为 400 Mbps、初始目标为 300 Mbps 时：

| Attempt | 目标 Mbps | BPS 百分比 |
| ---: | ---: | ---: |
| 1 | 300 | 75% |
| 2 | 240 | 60% |
| 3 | 180 | 45% |
| 4 | 120 | 30% |
| 5 | 60 | 15% |
| 6 | 30 | 7.5% |

## 时序数据与 DUT 采样

- **BPS 时序数据**：从 BPS 报告中提取 Tx/Rx、Concurrent Flows 和 Flow Rate，按最近采样点对齐到 1 秒时间轴。程序根据 Concurrent Flows 划分 Ramp-up、Stable 和 Ramp-down，并用 Stable 开始后的前 5 个样本冻结基线；Stable 阶段约 10% 的持续下降判为异常，约 20% 的持续下降判为严重异常，Ramp-down 中吞吐随负载下降视为正常变化。
- **DUT 前端采集**：打流前后分别采集设备、接口和系统快照；打流结束并等待冷却后，一次性读取前端 API 已记录的 CPU、内存、会话和接口流量历史数据，校正设备时钟后切分为 Baseline、During 和 Recovery 三个阶段。打流期间仅维持登录会话，不由本程序定时轮询指标。
- **DUT 后端采集（默认）**：打流期间由独立线程按 `dut.backend.interval_seconds` 定时通过 SSH 读取 CPU、内存、会话和指定接口流量。耗时过长而错过的周期直接跳过并计入 `missed_sample_count`；完整采样和失败记录写入 JSON，紧凑时序写入 CSV。

## 产物与判定

每次 Attempt 的产物包括：

- `bps-report.csv`：测试参数、判据和结果，内容进入 `evidence.json`。
- `bps-report-full.pdf`：使用 `-f` 时生成的 best-effort BPS 完整 PDF 结果报告；关键 CSV 导出完成后，
  由独立进程使用自己的 BPS Session 下载，不进入 LLM Evidence。任务状态写入
  `bps-report-full.job.json`，慢速下载不会阻塞 Evidence、Verdict 或 finalize。
- `bps-launch.json`：BPS 外部 Run 的 durable launch journal；Resume 使用它接管已启动
  Run，无法唯一 reconciliation 时停止为 `INCONCLUSIVE`，不会盲目重复打流。
- `bps-performance-timeseries.csv`：原始秒级性能数据，不进入 `evidence.json`。
- `dut-metrics.json`：使用 SSH 后端方式时，保存逐次采样的完整审计数据和失败记录，不发送给 LLM。
- `dut-metrics.csv`：使用 SSH 后端方式时，保存打流期间的紧凑 DUT 时序，仅首行保留真实时间锚点，后续行使用相对秒数；正文进入 `evidence.json` 交给 LLM。
- `evidence.json`：BPS 报告、紧凑的 `bps_performance_analysis`，以及启用时的 DUT 证据。
- `verdict.json`：解析后的裁决、provider response、模型参数以及 Evidence 路径和 SHA-256；
  不重复保存完整 Evidence 请求。
- TOC、Section 解析结果和 Attempt 审计文件。

最终 Outcome：

- `PASSED`：首次目标流量的 Attempt 被 LLM 判定为 `pass`。
- `DEGRADED_PASS`：首次目标未通过，但某个降载 Attempt 被判定为 `pass`。
- `NOT_PASSED`：六次 Attempt 均返回 `retry`。
- `INCONCLUSIVE`：证据不完整或外部接口失败。

CLI 退出码是稳定接口：

| Exit Code | 含义 |
| ---: | --- |
| `0` | `PASSED`，目标带宽首次通过 |
| `1` | `DEGRADED_PASS`，降载后通过 |
| `2` | `NOT_PASSED`，所有尝试均未通过 |
| `3` | `INCONCLUSIVE`，Evaluation 已执行但无法可靠裁决 |
| `4` | 配置、凭据、BPS/DUT/LLM、存储等运行错误 |
| `64` | CLI 参数使用错误 |
| `130` | 用户通过 Ctrl+C 中断 |

退出码只表示大的结果类别，具体故障原因由 `[ERROR_CODE]` 输出说明。自动化脚本应优先依据退出码判断 Evaluation 结果。

## 开发检查

```powershell
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m pytest
```

自动化测试使用仿真 BPS、DUT 和 LLM，不连接实验室设备。
