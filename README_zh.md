# Robot Agent 文档

本仓库库旨在通过一套工具和服务，为控制机器人提供全面的解决方案。它使用模型上下文协议（MCP）与机器人控制服务进行通信，从而实现高效、灵活的机器人操作。本 README 将指导您了解该资源库的主要组件、如何设置环境以及如何使用所提供的功能。

## 仓库结构

以下是仓库中主要文件和目录的概览：

- **`agent.py`**： 主代理脚本，用于处理用户信息，并使用 MCP 与机器人控制服务交互。
- **`agent_graph.py`**： 定义代理的图结构，包括处理不同任务的子图和节点。目前暂时处于调试阶段；
- **`proto/`**： 包含用于机器人控制和摄像头服务的 gRPC 协议缓冲区定义文件（`robot_control.proto`）和生成的 Python 代码（`robot_control_pb2.py`, `robot_control_pb2_grpc.py`）。
- **`robot_mcp_client.py`**： 用于与 MCP 服务器交互的异步客户端脚本。该脚本仅用于测试；
- **`robot_mcp_server.py`**： MCP 服务器脚本，用于连接基于 gRPC 的机器人控制服务。
- **`simulator.py`**： 模拟器脚本，用于启动 gRPC 服务器，模拟机器人控制和摄像头服务。
- **`prompts.py`**： 定义机器人控制代理的系统提示（system prompt）。
- **`.gitignore`**： 指定 Git 将忽略的文件和目录。
- **LICENSE**： 包含项目的许可证信息。

## 安装依赖

- **Python**： 本项目使用 Python 编写。建议使用 Python 3.10 或更高版本。

- **依赖关系**： 运行以下命令安装必要的 Python 软件包：

  ```shell
  pip install -r requirements.txt
  ```

## 项目启动与快速开始

### 1. 环境变量

您可以使用环境变量配置机器人控制服务的 gRPC 主机和端口：

- `ROBOT_GRPC_HOST`： gRPC 服务器的主机地址。默认为 `localhost`。
- `ROBOT_GRPC_PORT`： gRPC 服务器的端口号。默认为 `50051`。

您还可以修改 `common/config.py` 中的配置来直接指定，例如：

```python
server_params = StdioServerParameters(
    command="python",
    args=["robot_mcp_server.py", "--stdio", "--grpc-host", "192.168.12.210"],
)
```

### 2. 模型配置

参考 `common/config.py` 中的配置，修改为你自己的 API Key 等模型信息：

```python
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.3,
    base_url="https://api.openai.com/api/v1",
    # 这里是无效的 API Key，需要替换为自己的信息
    api_key="6666666666666666666"
)
```

### 3. 项目启动

首先想要执行项目需要导入项目包为本地包：

```shell
pip install -e .
```

（可选）然后开发时在 IDE 中设置：

- VSCode：在 `.vscode/settings.json` 添加：
  ```json
  {
    "terminal.integrated.env.linux": {"PYTHONPATH": "${workspaceFolder}"},
    "python.analysis.extraPaths": ["${workspaceFolder}"]
  }
  ```
  
- PyCharm：右键项目根目录 -> Mark Directory as -> Sources Root；



接下来，你需要先启动一个 HTTP server，接收来自手机/其他 agent 的消息，执行：

```shell
python servers/external_control.py
```

你可以在该文件中修改启动配置（例如端口、监听主机地址等等）；

如果你希望在模拟器中运行测试，则可以修改 `common/config.py` 中的监听地址为本地，并且启动 `servers/simulator.py`，即可下达指令！

> 注：直接运行 `py_agent/robot_agent.py` 即可使用命令行（而非 HTTP 接口）来与 Agent 交互。
