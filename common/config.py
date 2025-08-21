from langchain_openai import ChatOpenAI
from mcp import StdioServerParameters

# ────────────────────────────────────────────────────────────────────────────
# Configurations
# ────────────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(
    # model_name="gpt-4o-mini",
    model_name="doubao-1.5-lite-32k-250115",
    temperature=0.3,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="2ca0a798-9c36-4eee-b9c1-8349ec3595a7")
    # base_url="http://localhost:17100",
    # api_key="sk-dwjeifjiewrpijepwjiw")
server_params = StdioServerParameters(
    command="python",
    args=["servers/robot_mcp_server.py", "--stdio", "--grpc-host", "192.168.43.152"],
    # env={"PYTHONPATH": "."},
    # args=["robot_mcp_server.py", "--stdio"]
)
