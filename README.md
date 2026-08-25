# LangGraph BPS 性能测试评估 Agent

这是一个真实设备优先的演示项目：运行指定的 Keysight BreakingPoint 性能模板，打流结束后一次读取 DUT 资源历史并按测试时间窗口过滤，将原始 BPS 报告和 DUT 监控记录交给 DeepSeek 判断。模型返回 `pass` 时立即结束；返回 `retry` 时按初始流量目标的 80%、60%、40%、20% 逐档降载重试，包含首次执行在内最多五次。

## 环境

项目按 Python 3.11+ 编写。本仓库已约定使用以下 Conda 环境：

```powershell
conda activate shixi
Set-Location E:\f_bps
python -m pip install -e .
```

该环境需要 `langgraph`、`langgraph-checkpoint-sqlite`、`httpx`、`pydantic` 和 `PyYAML`。开发检查还使用 `pytest`、`ruff` 和 `mypy`。

需要运行全部开发检查时安装开发依赖：

```powershell
python -m pip install -e '.[dev]'
```

## 配置与凭据

复制并修改 `config/demo.yaml`。配置中只保存设备地址、模板、端口、接口、时序和模型名称；不要写入密码或 token。

以下 6 个条目保存在 Windows Credential Manager（Python `keyring` 服务名为 `nsfocus-bps-evaluation-agent`）：

- `BPS_USERNAME`、`BPS_PASSWORD`
- `DUT_USERNAME`、`DUT_PASSWORD`
- 公司中转：`COMPANY_DEEPSEEK_API_KEY`
- DeepSeek 官方：`DEEPSEEK_API_KEY`

一次性录入全部条目：

```powershell
python -m bps_agent credentials set
```

也可以只录入本次需要的条目：

```powershell
python -m bps_agent credentials set BPS_USERNAME BPS_PASSWORD DUT_USERNAME DUT_PASSWORD COMPANY_DEEPSEEK_API_KEY
```

检查保存状态，不会显示实际值：

```powershell
python -m bps_agent credentials status
```

删除指定条目或全部条目：

```powershell
python -m bps_agent credentials delete BPS_PASSWORD
python -m bps_agent credentials delete
```

运行时查找顺序为当前进程环境变量、Windows keyring、交互输入。交互输入的新值会自动写入 keyring；环境变量只作临时覆盖，不会自动复制进 keyring。DUT CAPTCHA 始终由操作者查看并输入，不会保存。认证材料不会进入 LangGraph checkpoint、日志或审计文件。

## 真实运行

以下命令均从项目根目录 `E:\f_bps` 运行。省略 `--config` 时默认读取 `config\demo.yaml`：

```powershell
python -m bps_agent run
```

`config/demo.yaml` 中的 `bps.total_bandwidth_mbps` 是首次打流的 Total Bandwidth 目标，默认 400 Mbps。Agent 在预检时调用 `getSharedComponentSettings`，解析返回 `result` 中 `totalBandwidth.originalValue` 作为当前模板的 100% 基准；不使用硬编码基准，也不使用可能已被上次操作修改的 `currentValue`。每次打流前再调用 `setSharedComponentSettings` 写入换算后的 `totalBandwidth` 百分比。

临时覆盖配置中的 BPS 模板名称、端口和首次流量目标：

```powershell
python -m bps_agent run --template other-performance-template --ports 4 5 --total-bandwidth-mbps 300
```

`--template` 必须与 BPS 中的模板名称完全一致；`--ports` 接受一个或多个端口号；`--total-bandwidth-mbps` 必须大于 0，且不能超过运行时从该模板 JSON 读取的 `originalValue`。覆盖值会写入本次 Evaluation 的配置快照。恢复已有 Evaluation 时不能使用这些覆盖参数。

首次 Attempt 使用配置的完整流量目标。若 LLM 返回 `retry`，后续四次分别使用首次目标的 80%、60%、40%、20%；例如模板 JSON 的 `originalValue` 为 400 Mbps、首次目标为 300 Mbps 时，五档目标为 300、240、180、120、60 Mbps，传给 BPS API 的百分比分别为 75、60、45、30、15。任一 Attempt 返回 `pass` 后立即结束，不再执行剩余档位。模板原始 Mbps、每次实际 Mbps 和 API 百分比都会写入 Attempt 与 Evidence 审计字段。

只执行实机打流与 Evidence 组装，并在调用 LLM 前停止：

```powershell
python -m bps_agent run --stop-before-llm
```

CLI 启动时先验证所选 DeepSeek 接口是否接受 JSON mode、thinking enabled 和 `reasoning_effort=max`，随后才登录 BPS/DUT 和执行真实打流。

命令行默认隐藏 `httpx` 的逐请求 INFO 日志，只显示项目流程、告警和错误。LLM 判定完成后会显示 Verdict 中的 `summary` 和 `observations`，完整响应仍保存在 `verdict.json`。

DUT 的 CPU、内存、会话和接口流量在每次 BPS Run 结束并冷却后各读取一次。Agent 使用打流前后系统时间校准 DUT 时钟，保留打流开始前 `baseline_seconds`（默认 600 秒）、打流期间和可用的恢复期数据点；没有新的恢复点不视为证据不完整。Evidence 将每个资源的原始响应元数据保留一次，并按资源组织 baseline、during、recovery 数据点，避免三个阶段重复响应外壳。接口状态、硬件健康和系统摘要仍在打流前后各读取一次。打流期间每隔 `keepalive_interval_seconds`（默认 60 秒）只读请求一次系统摘要以保持 DUT 会话；保活结果不写入 Evidence，单次失败只记录 Attempt 告警且不中断 BPS。`dut.period` 可在确认设备支持的取值后限制历史查询范围；省略时使用 DUT 默认范围。

BPS 报告导出前，Agent 会读取当前 BPS Run 的 TOC，按“章节标题 + 父级路径”动态解析本次实际的 Section ID，再调用 `exportReport`。因此 Section 编号随模板变化时无需修改配置。测试参数、判据和结果摘要导出为 `bps-report.csv` 并写入 Evidence；带秒级时间戳的整体收发吞吐率、Flow 建立速率和并发 Flow 数通过第二次 `exportReport` 单独导出为 `bps-performance-timeseries.csv`，用于独立分析性能波动，不嵌入 `evidence.json`，也不发送给 LLM。任一必需章节缺失会使本次 Evaluation 标记为 `INCONCLUSIVE`。

Agent 会对独立时序 CSV 做确定性分析：使用最近时间点映射到 1 秒时间轴，根据并发 Flow 识别 Ramp-up、Stable 和 Ramp-down，并以 Stable 阶段 Tx/Rx 中位数作为本次测试自己的吞吐基线。Stable 负载下持续 3 个采样周期的约 10% 下降判为性能异常，持续 2 个周期的约 20% 下降或持续扩大判为严重性能异常；Ramp-down 中吞吐与并发 Flow 同向下降判为正常负载变化。Flow Rate 仅作辅助证据，不单独触发异常。`evidence.json` 只保存 `bps_performance_analysis` 的阶段、基线、事件、阈值和最终分类，不保存原始逐秒数据或时序 CSV 路径；确定性分类供 LLM 裁决参考，不直接覆盖 Verdict。

进程输出 Evaluation Run ID。恢复已有 checkpoint：

```powershell
python -m bps_agent run --resume <evaluation-id>
```

当 BPS 运行状态或端口归属不明确时，程序保留本地端口组锁并停止自动清理。此时应先在 BPS 上人工核对运行和端口状态。

## 离线回放

```powershell
python -m bps_agent replay --evidence artifacts\<evaluation-id>\attempt-01\evidence.json
```

回放只读取已保存的 Evidence Bundle 并调用当前选择的 DeepSeek 接口，不访问 BPS 或 DUT。

## 结果与审计

Evaluation Run 的最终 Outcome：

- `PASSED`：某次完整 Attempt 被 LLM 判定为 `pass`；
- `NOT_PASSED`：五次完整 Attempt 均被判定为 `retry`；
- `INCONCLUSIVE`：证据不完整、外部接口失败或无法获得有效 Verdict。

默认审计目录为 `artifacts/`，SQLite checkpoint 和端口组锁位于 `.state/`。这些运行产物均被 Git 忽略。

`bps-report-toc.json` 保留本次导出章节筛选所依据的原始 TOC；`bps-report-sections.json` 和 `bps-performance-timeseries-sections.json` 分别保留主报告与性能时序报告的标题、父级路径和动态 Section ID 解析结果。这些审计文件都不嵌入 Evidence，也不发送给 LLM。

## 开发检查

```powershell
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m pytest
```

自动化测试通过应用级 Evaluation Run 接缝注入仿真 BPS、DUT 和 LLM，不连接实验室设备。
