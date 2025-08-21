# Robot Agent Documentation

[中文文档](./README_zh.md)

## Introduction

The `robot-agent` repository is designed to provide a comprehensive solution for  controlling robots through a set of tools and services. It uses the  Model Context Protocol (MCP) to communicate with robot control services, enabling efficient and flexible robot operation. This README will guide you through the main components of the repository, how to set up the  environment, and how to use the provided functionality.

## Repository Structure

Here is an overview of the main files and directories in the repository:

- **`agent.py`**: The main agent script that processes user messages and interacts with the robot control services using the MCP.
- **`agent_graph.py`**: Defines the graph structure for the agent, including the sub - graphs and nodes for handling different tasks.
- **`proto/`**: Contains the Protocol Buffers (gRPC standard) definition files (`robot_control.proto`) and the generated Python code (`robot_control_pb2.py`, `robot_control_pb2_grpc.py`) for the robot control and camera services.
- **`robot_mcp_client.py`**: An asynchronous client script for interacting with the MCP server.
- **`robot_mcp_server.py`**: The MCP server script that bridges the gRPC - based robot control services.
- **`simulator.py`**: A simulator script for starting a gRPC server to simulate robot control and camera services.
- **`prompts.py`**: Defines the system prompt for the robot control agent.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`LICENSE`**: Contains the license information for the project.

## Prerequisites

- **Python**: This project is written in Python. It is recommended to use Python 3.10 or higher.
- **Dependencies**: Install the necessary Python packages by running the following command:

```bash
pip install -r requirements.txt
```

Note: The `requirements.txt` file is not provided in the given code snippets. You need to create it based on the imports in the Python files, such as `langchain_openai`, `langgraph`, `fastmcp`, `grpcio`, etc.

## Project startup and quick start

### 1. Environment Variables

You can use environment variables to configure the gRPC host and port for the robot control service:

- `ROBOT_GRPC_HOST`: Host address of the gRPC server. The default is `localhost`.
- `ROBOT_GRPC_PORT`: Port number of the gRPC server. The default is `50051`.

You can also modify the configuration in Agent (`common/config.py`) to specify it directly, for example:

``` python 
server_params = StdioServerParameters(
    command="python",
    args=["robot_mcp_server.py", "--stdio", "--grpc-host", "192.168.12.210"],
)
```

### 2. Model Configuration

Refer to the configuration in ``common/config.py`` and modify it to your own model information such as API Key:

``` python 
llm = ChatOpenAI(
    model_name="gpt-4o",
    temperature=0.3,
    base_url="https://api.openai.com/api/v1",
    ## This is an invalid API Key, you need to replace it with your own information
    api_key="6666666666666666666666666"
)
```

### 3. Starting the project

First of all to execute the project you need to import the project package as a local package:

```shell 
pip install -e .
```

(optional) and then set it up in the IDE during development:

- VSCode: add in `.vscode/settings.json`: 
    ```json 
    {
        "terminal.integrated.env.linux": {"PYTHONPATH": "${workspaceFolder}"},
        "python.analysis.extraPaths": ["${workspaceFolder}"]
    }
    ```
  
- PyCharm: Right click on the project root -> Mark Directory as -> Sources Root;



Next, you need to start an HTTP server first to receive messages from your phone/other agent to execute:

```shell 
python servers/external_control.py 
```

You can modify the startup configuration (e.g. ports, listening host addresses, etc.) in that file;

If you want to run the tests in the simulator, you can change the listening address in `common/config.py` to local and start `servers/simulator.py` to give commands!

> Note: Running `py_agent/robot_agent.py` directly will allow you to interact with Agent using the command line (not the HTTP interface).

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contributing

If you want to contribute to this project, please fork the repository, make your changes, and submit a pull request.

## Support

If you encounter any issues or have questions, please open an issue on the GitHub repository.

