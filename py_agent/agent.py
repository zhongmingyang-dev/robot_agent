
from typing import AsyncGenerator, Any, Tuple


class Agent:
    def submit_message(
            self,
            message: str,
            request_id: str | None = None,
            timeout: float | None = None
    ) -> Tuple[Any, str]:
        """
        Synchronous message submission and get full response
        :param message: user input message
        :param request_id: request identifier
        :param timeout: processing timeout
        :return: full response or error message
        """
        raise NotImplementedError("Agent::submit_message not implemented")

    async def submit_message_async(
            self,
            message: str,
            request_id: str | None = None,
            timeout: float | None = None
    ) -> AsyncGenerator[Tuple[Any, str], None]:
        """
        Asynchronous submission message
        :param message: user input message
        :param request_id: request identifier
        :param timeout: processing timeout
        :yield: (response type, content)
        """
        raise NotImplementedError("Agent::submit_message_async not implemented")
