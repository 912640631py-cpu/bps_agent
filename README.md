# BPS 性能测试评估 Agent

基于 LangGraph 的 Keysight BreakingPoint 实机测试工具。它负责 BPS 打流、可选的 DUT 监控、报告导出、性能波动分析，并由 DeepSeek 给出最终 Verdict。

## 主要能力

- 从每个 Run 的 TOC 按“章节标题 + 父级路径”动态解析 Section ID。
- 分别导出主报告和秒级吞吐、Flow Rate、Concurrent Flows 时序数据。
- 在 Stable 阶段以 Tx/Rx 中位数为基线进行确定性性能分析。
- LLM 返回 `retry` 时自动降载，最多执行五次；返回 `pass` 后立即结束。
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
- `DUT_USERNAME`、`DUT_PASSWORD`（默认前端采集）
- `DUT_BACKEND_USERNAME`、`DUT_BACKEND_PASSWORD`（仅 SSH 后端采集）
- `COMPANY_DEEPSEEK_API_KEY` 或 `DEEPSEEK_API_KEY`

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
python -m bps_agent run --template TEMPLATE --ports 4 5 --total-bandwidth-mbps 300

# 临时覆盖 DUT SSH 后端目标、接口和采样间隔
python -m bps_agent run --dut-collection-method backend_ssh `
  --dut-host 10.66.246.156 --dut-port 50023 `
  --dut-interface T1/1 --dut-interface T1/2 --dut-interval-seconds 10

# 完成实机测试和 Evidence，在调用 LLM 前停止
python -m bps_agent run --stop-before-llm

# 只使用 BPS，不登录或读取 DUT
python -m bps_agent run --bps-only

# 恢复已有 Evaluation
python -m bps_agent run --resume EVALUATION_ID

# 使用已有 Evidence 重新裁决
python -m bps_agent replay --evidence artifacts\EVALUATION_ID\attempt-01\evidence.json
```

恢复运行时不能使用模板、端口、带宽、DUT 或模式覆盖参数。
也可以在 YAML 中设置 `evaluation.mode: bps_only`；默认值为 `bps_and_dut`。
`dut.collection_method` 默认是 `frontend_api`，CAPTCHA 由操作者输入且不会保存。
设置为 `backend_ssh` 后，采样从 BPS 打流开始持续到流量结束，不使用固定样本数，
也不登录 DUT 前端或请求 CAPTCHA。当前 SSH 方式不读取、校验或持久化主机密钥。
端口互斥由 BPS 的非强制预留负责；项目不再维护本地端口锁文件。

## 降载策略

首次 Attempt 使用配置目标。若 LLM 判定未通过，后续目标依次为初始目标的 `80%`、`60%`、`40%`、`20%`。

例如模板原始带宽为 400 Mbps、初始目标为 300 Mbps 时：

| Attempt | 目标 Mbps | BPS 百分比 |
| ---: | ---: | ---: |
| 1 | 300 | 75% |
| 2 | 240 | 60% |
| 3 | 180 | 45% |
| 4 | 120 | 30% |
| 5 | 60 | 15% |

## 产物与判定

每次 Attempt 默认生成：

- `bps-report.csv`：测试参数、判据和结果，内容进入 `evidence.json`。
- `bps-performance-timeseries.csv`：原始秒级性能数据，不进入 `evidence.json`。
- `dut-metrics.json`：使用 SSH 后端方式时，保存逐次采样的完整审计数据和失败记录，不发送给 LLM。
- `dut-metrics.csv`：使用 SSH 后端方式时，保存打流期间的紧凑 DUT 时序，正文进入 `evidence.json` 交给 LLM。
- `evidence.json`：BPS 报告、紧凑的 `bps_performance_analysis`，以及启用时的 DUT 证据。
- TOC、Section 解析结果、Attempt 和 Verdict 审计文件。

性能时序按最近采样点对齐到 1 秒时间轴。Stable 负载下，吞吐下降超过约 10% 且持续至少 3 秒判为性能异常；下降超过约 20% 且持续至少 2 秒判为严重异常。Ramp-down 中吞吐与 Concurrent Flows 同步下降视为正常负载变化。

最终 Outcome：

- `PASSED`：首次目标流量的 Attempt 被 LLM 判定为 `pass`。
- `DEGRADED_PASS`：首次目标未通过，但某个降载 Attempt 被判定为 `pass`。
- `NOT_PASSED`：五次 Attempt 均返回 `retry`。
- `INCONCLUSIVE`：证据不完整或外部接口失败。

## 开发检查

```powershell
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m pytest
```

自动化测试使用仿真 BPS、DUT 和 LLM，不连接实验室设备。
