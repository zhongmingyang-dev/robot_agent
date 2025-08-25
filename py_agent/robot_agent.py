import signal
import sys

from py_agent.agent import Agent

from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from mcp import ClientSession, StdioServerParameters, McpError
from mcp.client.stdio import stdio_client

import asyncio
import logging
from typing import AsyncGenerator, Tuple, Set

import threading
import queue
import time

from dataclasses import dataclass
from enum import Enum

from common import prompts, config


class AgentErrorCode(Enum):
    SUCCESS = 0
    TIMEOUT = 1
    UNKNOWN = 2


@dataclass(frozen=True)
class AgentRecvQueueMessage:
    code: AgentErrorCode
    msg: str
    req_id: str = "[DEFAULT-REQ-ID]"


@dataclass(frozen=True)
class AgentSendQueueMessage:
    msg: str
    resp_queue: queue.Queue
    req_id: str = "[DEFAULT-REQ-ID]"


class RobotAgent(Agent):
    def __init__(
            self,
            vlm: ChatOpenAI,
            mcp_server_params: StdioServerParameters,
            system_message: str = "You are a helpful assistant",
            log_level: int = logging.DEBUG
    ):
        self.vlm = vlm
        self.mcp_server_params = mcp_server_params
        self.system_message = system_message
        self.log_level = log_level

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.logger.setLevel(self.log_level)
        ch.setLevel(self.log_level)

        self._is_running = False
        self._loop: asyncio.AbstractEventLoop | None = None  # 服务事件循环
        self._thread: threading.Thread | None = None  # 服务线程
        self._session: ClientSession | None = None
        self._agent = None
        self._tools: BaseTool | None = None

        # 线程安全的同步队列
        self._message_queue = queue.Queue[AgentSendQueueMessage]()
        self._response_queues: Set[AgentRecvQueueMessage] = set()

        self.logger.info("Agent service initialized")

    def start(self):
        """启动服务线程"""
        if self._is_running:
            return

        self._is_running = True
        self._thread = threading.Thread(target=self._run_service, daemon=True)
        self._thread.start()
        self.logger.info("Service thread started")

    def stop(self):
        """停止服务线程"""
        if not self._is_running:
            return

        self._is_running = False
        if self._thread:
            self._thread.join()
            self._thread = None
        self.logger.info("Service thread stopped")

    def _run_service(self):
        """服务线程主函数"""
        # 创建服务事件循环
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        try:
            self._loop.run_until_complete(
                self._service_main(),
            )
        except Exception as e:
            print("❌ 异常:", repr(e))
        finally:
            self._loop.close()
            self._loop = None

    async def _service_main(self):
        """服务主协程"""
        self.logger.debug("Starting agent service...")
        try:
            async with stdio_client(self.mcp_server_params) as (read, write):
                self.logger.debug("Agent MCP connection established")
                async with ClientSession(read, write) as self._session:
                    await self._session.initialize()
                    self.logger.debug("Agent MCP session initialized")

                    self._tools = await load_mcp_tools(self._session)
                    self.logger.debug(f"Agent loaded {len(self._tools)} tools")

                    self._agent = create_react_agent(self.vlm, self._tools)
                    self.logger.debug("Agent created")

                    # 主消息处理循环
                    while self._is_running and await self._session.send_ping():
                        try:
                            # 设置 timeout 用于定期检查 _is_running 以及 MCP server 连接状态
                            pkt = self._message_queue.get(block=True, timeout=3.0)
                            request_id = pkt.req_id
                            self.logger.info(f"[{request_id}] Processing message")
                            await self._process_message(pkt)
                        except queue.Empty:
                            continue
                        except Exception as e:
                            self.logger.exception(f"Error in service main: {str(e)}")
                            await asyncio.sleep(1)
        except asyncio.CancelledError:
            self.logger.warning("Agent service stopped due to cancellation")
        except McpError:
            self.logger.warning("MCP server is dead. Agent service stopped")
        finally:
            self._session = None
            self._agent = None
            self._tools = None

    async def _process_message(self, pkt: AgentSendQueueMessage):
        """处理单个消息"""
        message, response_queue, request_id = pkt.msg, pkt.resp_queue, pkt.req_id
        try:
            # 准备消息
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": message}
            ]

            self.logger.debug(f"[{request_id}] Sending to agent")
            async for step in self._agent.astream(
                    {"messages": messages},
                    stream_mode="values"
            ):
                # 获取最新消息内容
                content_msg_obj = step["messages"][-1]
                if content_msg_obj:
                    content: str = content_msg_obj.pretty_repr()
                    self.logger.debug(f"[{request_id}] Agent response chunk:\n{content}")
                    resp = AgentRecvQueueMessage(AgentErrorCode.SUCCESS, content, request_id)
                    from servers.external_control import send_to_user
                    send_to_user(resp.msg)
                    # 将响应放入同步队列
                    response_queue.put(resp)

            # None 标记流结束
            self.logger.debug(f"[{request_id}] End of agent stream")
            response_queue.put(None)

        except Exception as e:
            self.logger.exception(f"[{request_id}] Error during message processing")
            resp = AgentRecvQueueMessage(AgentErrorCode.UNKNOWN, f"Processing error: {str(e)}", request_id)
            response_queue.put(resp)

    def submit_message(
            self,
            message: str,
            request_id: str | None = None,
            timeout: float | None = None
    ) -> Tuple[AgentErrorCode, str]:
        """
        同步提交消息并获取完整响应
        :param message: 用户输入消息
        :param request_id: 请求标识符
        :param timeout: 处理超时时间
        :return: 完整响应或错误信息
        """

        if not self._is_running:
            return AgentErrorCode.UNKNOWN, "Agent not running"

        request_id = request_id or f"req-{threading.get_ident()}-{id(message)}"
        self.logger.info(f"[{request_id}] Submitting message: {message[:50]}...")

        # 创建响应队列
        response_queue = queue.Queue()

        # 将消息放入服务队列
        pkt = AgentSendQueueMessage(message, response_queue, request_id)
        self._message_queue.put(pkt)

        full_response = []
        start_time = time.time()

        try:
            while True:
                try:
                    rcv_pkt: AgentRecvQueueMessage | None = response_queue.get(timeout=timeout)
                except queue.Empty:
                    self.logger.warning(f"[{request_id}] Response timeout after {timeout}s")
                    return AgentErrorCode.TIMEOUT, "Response timed out"

                if rcv_pkt is None:
                    # Stream end
                    self.logger.info(f"[{request_id}] Response completed")
                    return AgentErrorCode.SUCCESS, "".join(full_response)
                elif rcv_pkt.code == AgentErrorCode.SUCCESS:
                    full_response.append(rcv_pkt.msg)
                else:
                    self.logger.error(f"[{request_id}] Error response with code {rcv_pkt.code}: {rcv_pkt.msg}")
                    return rcv_pkt.code, rcv_pkt.msg

                # 检查流式传输总超时
                if timeout is not None and time.time() - start_time > timeout:
                    self.logger.warning(f"[{request_id}] Overall timeout after {timeout}s")
                    return AgentErrorCode.TIMEOUT, "Overall timeout"

        except Exception as e:
            self.logger.exception(f"[{request_id}] Unexpected error during response handling")
            return AgentErrorCode.UNKNOWN, f"Internal error: {str(e)}"

    async def submit_message_async(
            self,
            message: str,
            request_id: str | None = None,
            timeout: float | None = None
    ) -> AsyncGenerator[Tuple[AgentErrorCode, str], None]:
        """
        异步提交消息（需要在服务线程中调用）
        :param message: 用户输入消息
        :param request_id: 请求标识符
        :param timeout: 处理超时时间
        :yield: (响应类型, 内容) 元组
        """
        if not self._is_running:
            yield AgentErrorCode.UNKNOWN, "Service not running"
            return

        if asyncio.get_running_loop() != self._loop:
            yield AgentErrorCode.UNKNOWN, "Must be called from service thread"
            return

        request_id = request_id or f"req-{id(message)}"
        self.logger.info(f"[{request_id}] Submitting message: {message[:50]}...")

        # 创建响应队列
        response_queue = queue.Queue()

        # 将消息放入服务队列
        pkt = AgentSendQueueMessage(message, response_queue, request_id)
        self._message_queue.put(pkt)

        start_time = time.time()

        try:
            while True:
                try:
                    rcv_pkt: AgentRecvQueueMessage | None = response_queue.get(timeout=timeout)
                except queue.Empty:
                    self.logger.warning(f"[{request_id}] Response timeout after {timeout}s")
                    yield AgentErrorCode.TIMEOUT, "Response timed out"
                    return

                if rcv_pkt is None:
                    # Stream end
                    self.logger.info(f"[{request_id}] Response completed")
                    return
                elif rcv_pkt.code == AgentErrorCode.SUCCESS:
                    yield AgentErrorCode.SUCCESS, rcv_pkt.msg
                else:
                    self.logger.error(f"[{request_id}] Error response with code {rcv_pkt.code}: {rcv_pkt.msg}")
                    yield rcv_pkt.code, rcv_pkt.msg
                    return

                # 检查流式传输总超时
                if timeout is not None and time.time() - start_time > timeout:
                    self.logger.warning(f"[{request_id}] Overall timeout after {timeout}s")
                    yield AgentErrorCode.TIMEOUT, "Overall timeout"
                    return

        except Exception as e:
            self.logger.exception(f"[{request_id}] Unexpected error during response handling")
            yield AgentErrorCode.UNKNOWN, f"Internal error: {str(e)}"


if __name__ == '__main__':
    agent = RobotAgent(config.llm, config.server_params, prompts.SYSTEM_PROMPT)


    def handle_signal(signum, _frame):
        print(f"\nReceived signal {signum}, stopping agent...")
        agent.stop()
        sys.exit(0)


    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        agent.start()
        while True:
            user = input("User> ")
            _code, _msg = agent.submit_message(user)
            if _code != AgentErrorCode.SUCCESS:
                print(f"[Agent Error] {_msg}")
            else:
                print(_msg)
            
    except KeyboardInterrupt:
        # 处理再次按Ctrl+C的情况
        handle_signal(signal.SIGINT, None)
