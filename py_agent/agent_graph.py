import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, StateGraph
from langgraph.constants import START, END
from langchain_openai import ChatOpenAI


model = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.3,
    base_url="http://localhost:17100",
    api_key="sk-dwjeifjiewrpijepwjiw")


class ComplicatedState(MessagesState):
    new_field: str


async def make_graph():
    client = MultiServerMCPClient(
        {
            # "web-search": {
            #     "url": "http://localhost:8000/sse",
            #     "transport": "sse",
            # },
            "robot-controller": {
                "command": "python",
                "args": ["robot_mcp_server.py", "--stdio"],
                "transport": "stdio"
            }
        }
    )
    # since get_tools is not an async, it could be called directly
    tools = await client.get_tools()

    sub_graph_workflow = StateGraph(ComplicatedState)
    sub_graph_workflow.add_node("executor", create_react_agent(model, tools))
    sub_graph_workflow.add_edge(START, "executor")
    sub_graph_workflow.add_edge("executor", END)
    sub_graph = sub_graph_workflow.compile()

    def call_sub_graph(state: MessagesState):
        # this is just a demonstration, the graph could be larger
        # there will be more invoke and more def instead of async def
        return sub_graph.invoke(ComplicatedState(messages=state["messages"], new_field="something"))

    main_graph_workflow = StateGraph(MessagesState)
    main_graph_workflow.add_node("sub_graph", call_sub_graph)
    main_graph_workflow.add_edge(START, "sub_graph")
    main_graph_workflow.add_edge("sub_graph", END)
    main_graph = main_graph_workflow.compile()

    return main_graph


async def main():
    graph = await make_graph()
    result = await graph.ainvoke({"messages": "What is the weather like in New York?"})
    print(result)

# Change the transport to 'sse' and start the mcp server first:
# https://github.com/modelcontextprotocol/quickstart-resources/tree/main/weather-server-python

if __name__ == "__main__":
    asyncio.run(main())
