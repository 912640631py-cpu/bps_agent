# LangGraph BPS 性能测试评估 Agent

这是一个真实设备优先的演示项目：运行指定的 Keysight BreakingPoint 性能模板，在打流期间采集 DUT 资源，将原始 BPS 报告和 DUT 监控记录交给 DeepSeek 判断。模型返回 `pass` 时结束；返回 `retry` 时使用相同配置重新打流，包含首次执行在内最多三次。

## 环境

项目按 Python 3.11+ 编写。本仓库已约定使用以下 Conda 环境：

```powershell
$taskPython = 'E:\program1\anaconda3\envs\shixi\python.exe'
& $taskPython -m pip install -e .
```

该环境需要 `langgraph`、`langgraph-checkpoint-sqlite`、`httpx`、`pydantic` 和 `PyYAML`。开发检查还使用 `pytest`、`ruff` 和 `mypy`。

需要运行全部开发检查时安装开发依赖：

```powershell
& $taskPython -m pip install -e '.[dev]'
```

## 配置与凭据

复制并修改 `config/demo.yaml`。配置中只保存设备地址、模板、端口、接口、时序和模型名称；不要写入密码或 token。

支持的环境变量：

- `BPS_USERNAME`、`BPS_PASSWORD`
- `DUT_USERNAME`、`DUT_PASSWORD`
- 公司中转：`COMPANY_DEEPSEEK_API_KEY`
- DeepSeek 官方：`DEEPSEEK_API_KEY`

缺少用户名、密码或 token 时，CLI 会交互询问。DUT CAPTCHA 始终由操作者查看并输入。认证材料不会进入 LangGraph checkpoint、日志或审计文件。

## 真实运行

```powershell
$taskPython = 'E:\program1\anaconda3\envs\shixi\python.exe'
& $taskPython -m bps_agent run --config config/demo.yaml
```

CLI 启动时先验证所选 DeepSeek 接口是否接受 JSON mode、thinking enabled 和 `reasoning_effort=max`，随后才登录 BPS/DUT 和执行真实打流。

进程输出 Evaluation Run ID。恢复已有 checkpoint：

```powershell
& $taskPython -m bps_agent run --config config/demo.yaml --resume <evaluation-id>
```

当 BPS 运行状态或端口归属不明确时，程序保留本地端口组锁并停止自动清理。此时应先在 BPS 上人工核对运行和端口状态。

## 离线回放

```powershell
& $taskPython -m bps_agent replay `
  --config config/demo.yaml `
  --evidence artifacts/<evaluation-id>/attempt-01/evidence.json
```

回放只读取已保存的 Evidence Bundle 并调用当前选择的 DeepSeek 接口，不访问 BPS 或 DUT。

## 结果与审计

Evaluation Run 的最终 Outcome：

- `PASSED`：某次完整 Attempt 被 LLM 判定为 `pass`；
- `NOT_PASSED`：三次完整 Attempt 均被判定为 `retry`；
- `INCONCLUSIVE`：证据不完整、外部接口失败或无法获得有效 Verdict。

默认审计目录为 `artifacts/`，SQLite checkpoint 和端口组锁位于 `.state/`。这些运行产物均被 Git 忽略。

## 开发检查

```powershell
& $taskPython -m ruff format --check src tests
& $taskPython -m ruff check src tests
& $taskPython -m mypy src
& $taskPython -m pytest
```

自动化测试通过应用级 Evaluation Run 接缝注入仿真 BPS、DUT 和 LLM，不连接实验室设备。
