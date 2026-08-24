# LangGraph BPS 性能测试评估 Agent

这是一个真实设备优先的演示项目：运行指定的 Keysight BreakingPoint 性能模板，打流结束后一次读取 DUT 资源历史并按测试时间窗口过滤，将原始 BPS 报告和 DUT 监控记录交给 DeepSeek 判断。模型返回 `pass` 时结束；返回 `retry` 时使用相同配置重新打流，包含首次执行在内最多三次。

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

```powershell
python -m bps_agent run --config config\demo.yaml
```

只执行实机打流与 Evidence 组装，并在调用 LLM 前停止：

```powershell
python -m bps_agent run --config config\demo.yaml --stop-before-llm
```

CLI 启动时先验证所选 DeepSeek 接口是否接受 JSON mode、thinking enabled 和 `reasoning_effort=max`，随后才登录 BPS/DUT 和执行真实打流。

DUT 的 CPU、内存、会话和接口流量在每次 BPS Run 结束并冷却后各读取一次。Agent 使用打流前后系统时间校准 DUT 时钟，保留打流开始前 `baseline_seconds`（默认 600 秒）、打流期间和可用的恢复期数据点；没有新的恢复点不视为证据不完整。Evidence 将每个资源的原始响应元数据保留一次，并按资源组织 baseline、during、recovery 数据点，避免三个阶段重复响应外壳。接口状态、硬件健康和系统摘要仍在打流前后各读取一次。打流期间每隔 `keepalive_interval_seconds`（默认 60 秒）只读请求一次系统摘要以保持 DUT 会话；保活结果不写入 Evidence，单次失败只记录 Attempt 告警且不中断 BPS。`dut.period` 可在确认设备支持的取值后限制历史查询范围；省略时使用 DUT 默认范围。

进程输出 Evaluation Run ID。恢复已有 checkpoint：

```powershell
python -m bps_agent run --config config\demo.yaml --resume <evaluation-id>
```

当 BPS 运行状态或端口归属不明确时，程序保留本地端口组锁并停止自动清理。此时应先在 BPS 上人工核对运行和端口状态。

## 离线回放

```powershell
python -m bps_agent replay --config config\demo.yaml --evidence artifacts\<evaluation-id>\attempt-01\evidence.json
```

回放只读取已保存的 Evidence Bundle 并调用当前选择的 DeepSeek 接口，不访问 BPS 或 DUT。

## 结果与审计

Evaluation Run 的最终 Outcome：

- `PASSED`：某次完整 Attempt 被 LLM 判定为 `pass`；
- `NOT_PASSED`：三次完整 Attempt 均被判定为 `retry`；
- `INCONCLUSIVE`：证据不完整、外部接口失败或无法获得有效 Verdict。

默认审计目录为 `artifacts/`，SQLite checkpoint 和端口组锁位于 `.state/`。这些运行产物均被 Git 忽略。

`bps-report-toc.json` 作为独立审计产物保留，不嵌入 Evidence，也不发送给 LLM。

## 开发检查

```powershell
python -m ruff format --check src tests
python -m ruff check src tests
python -m mypy src
python -m pytest
```

自动化测试通过应用级 Evaluation Run 接缝注入仿真 BPS、DUT 和 LLM，不连接实验室设备。
