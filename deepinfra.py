"""deepinfra.com chat models wrapper"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import (Any, AsyncIterator, Callable, Dict, Iterator, List,
                    Mapping, Optional, Tuple, Type, Union)

import aiohttp
import requests
from langchain_community.utilities.requests import Requests
from langchain_core.callbacks.manager import (AsyncCallbackManagerForLLMRun,
                                              CallbackManagerForLLMRun)
from langchain_core.language_models.chat_models import (BaseChatModel,
                                                        agenerate_from_stream,
                                                        generate_from_stream)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (AIMessage, AIMessageChunk, BaseMessage,
                                     BaseMessageChunk, ChatMessage,
                                     ChatMessageChunk, FunctionMessage,
                                     FunctionMessageChunk, HumanMessage,
                                     HumanMessageChunk, SystemMessage,
                                     SystemMessageChunk)
from langchain_core.outputs import (ChatGeneration, ChatGenerationChunk,
                                    ChatResult)
from langchain_core.pydantic_v1 import Field

__all__ = ["ChatDeepInfra"]


logger = logging.getLogger(__name__)


class ChatDeepInfraException(Exception):
    """Exception raised when the DeepInfra API returns an error."""

    pass


def _create_retry_decorator(
    llm: ChatDeepInfra,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Returns a tenacity retry decorator, preconfigured to handle PaLM exceptions."""
    return create_base_retry_decorator(
        error_types=[requests.exceptions.ConnectTimeout, ChatDeepInfraException],
        max_retries=llm.max_retries,
        run_manager=run_manager,
    )


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    role = _dict["role"]
    if role == "user":
        return HumanMessage(content=_dict["content"])
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        if _dict.get("function_call"):
            additional_kwargs = {"function_call": dict(_dict["function_call"])}
        else:
            additional_kwargs = {}
        return AIMessage(content=content, additional_kwargs=additional_kwargs)
    elif role == "system":
        return SystemMessage(content=_dict["content"])
    elif role == "function":
        return FunctionMessage(content=_dict["content"], name=_dict["name"])
    else:
        return ChatMessage(content=_dict["content"], role=role)


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    if _dict.get("function_call"):
        additional_kwargs = {"function_call": dict(_dict["function_call"])}
    else:
        additional_kwargs = {}

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


def _convert_message_to_dict(message: BaseMessage) -> dict:
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise ValueError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict


class ChatDeepInfra(BaseChatModel):
    """A chat model that uses the DeepInfra API."""

    # client: Any  #: :meta private:
    model_name: str = Field(default="mistralai/Mixtral-8x7B-Instruct-v0.1", alias="model")
    """Model name to use."""
    #### !! ADD deepinfra_api_token !! ####
    deepinfra_api_token: Optional[str] = ''
    request_timeout: Optional[float] = Field(default=600, alias="timeout")
    request_sleep: Optional[float] = Field(default=2, alias="sleep")
    temperature: Optional[float] = 0.7
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Run inference with this temperature. Must by in the closed
       interval [0.0, 1.0]."""
    top_p: Optional[float] = 0.9
    """Decode using nucleus sampling: consider the smallest set of tokens whose
       probability sum is at least top_p. Must be in the closed interval [0.0, 1.0]."""
    top_k: Optional[int] = 0
    """Decode using top-k sampling: consider the set of top_k most probable tokens.
       Must be positive."""
    n: int = 1
    """Number of chat completions to generate for each prompt. Note that the API may
       not return the full n completions if duplicates are generated."""
    max_tokens: int = 1024
    streaming: bool = False
    max_retries: int = 1

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream": self.streaming,
            "n": self.n,
            "temperature": self.temperature,
            "request_timeout": self.request_timeout,
            **self.model_kwargs,
        }

    @property
    def _client_params(self) -> Dict[str, Any]:
        """Get the parameters used for the openai client."""
        return {**self._default_params}

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            try:
                request_timeout = kwargs.pop("request_timeout")
                request = Requests(headers=self._headers())
                response = request.post(
                    url=self._url(), data=self._body(kwargs), timeout=request_timeout
                )
                time.sleep(self.request_sleep)
                self._handle_status(response.status_code, response.text)
                return response
            except Exception as e:
                logger.critical(e)  # noqa: T201
                raise

        return _completion_with_retry(**kwargs)

    async def acompletion_with_retry(
        self,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use tenacity to retry the async completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            try:
                request_timeout = kwargs.pop("request_timeout")
                request = Requests(headers=self._headers())
                async with request.apost(
                    url=self._url(), data=self._body(kwargs), timeout=request_timeout
                ) as response:
                    self._handle_status(response.status, response.text)
                    return await response.json()
                await asyncio.sleep(self.request_sleep)
            except Exception as e:
                logger.critical(e)  # noqa: T201
                raise

        return await _completion_with_retry(**kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs}
        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        return self._create_chat_result(response.json())

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=res.get("finish_reason")),
            )
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model": self.model_name}
        res = ChatResult(generations=generations, llm_output=llm_output)
        return res

    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        params = self._client_params
        if stop is not None:
            if "stop" in params:
                raise ValueError("`stop` found in both the input and default params.")
            params["stop"] = stop
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        return message_dicts, params

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {**params, **kwargs, "stream": True}

        response = self.completion_with_retry(
            messages=message_dicts, run_manager=run_manager, **params
        )
        for line in _parse_stream(response.iter_lines()):
            chunk = _handle_sse_line(line)
            if chunk:
                yield ChatGenerationChunk(message=chunk, generation_info=None)
                if run_manager:
                    run_manager.on_llm_new_token(str(chunk.content))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {"messages": message_dicts, "stream": True, **params, **kwargs}

        request_timeout = params.pop("request_timeout")
        request = Requests(headers=self._headers())
        async with request.apost(
            url=self._url(), data=self._body(params), timeout=request_timeout
        ) as response:
            async for line in _parse_stream_async(response.content):
                chunk = _handle_sse_line(line)
                if chunk:
                    yield ChatGenerationChunk(message=chunk, generation_info=None)
                    if run_manager:
                        await run_manager.on_llm_new_token(str(chunk.content))
        await asyncio.sleep(self.request_sleep)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        message_dicts, params = self._create_message_dicts(messages, stop)
        params = {"messages": message_dicts, **params, **kwargs}

        res = await self.acompletion_with_retry(run_manager=run_manager, **params)
        return self._create_chat_result(res)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "n": self.n,
        }

    @property
    def _llm_type(self) -> str:
        return "deepinfra-chat"

    def _handle_status(self, code: int, text: Any) -> None:
        if code >= 500:
            raise ChatDeepInfraException(f"DeepInfra Server: Error {code}")
        elif code >= 400:
            raise ValueError(f"DeepInfra received an invalid payload: {text}")
        elif code != 200:
            raise Exception(
                "DeepInfra returned an unexpected response with status "
                f"{code}: {text}"
            )

    def _url(self) -> str:
        return "https://api.deepinfra.com/v1/openai/chat/completions"

    def _headers(self) -> Dict:
        return {
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
            "Authorization": f"Bearer {self.deepinfra_api_token}",
            "Connection": "keep-alive",
            "Content-Type": "application/json",
            "Host": "api.deepinfra.com",
            "Origin": "https://deepinfra.com",
            "Referer": "https://deepinfra.com/",
            "Sec-Ch-Ua": '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"""',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"macOS"',
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-site",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
            "X-Deepinfra-Source": "model-embed",
        }

    def _body(self, kwargs: Any) -> Dict:
        return kwargs


def _parse_stream(rbody: Iterator[bytes]) -> Iterator[str]:
    for line in rbody:
        _line = _parse_stream_helper(line)
        if _line is not None:
            yield _line


async def _parse_stream_async(rbody: aiohttp.StreamReader) -> AsyncIterator[str]:
    async for line in rbody:
        _line = _parse_stream_helper(line)
        if _line is not None:
            yield _line


def _parse_stream_helper(line: bytes) -> Optional[str]:
    if line and line.startswith(b"data:"):
        if line.startswith(b"data: "):
            # SSE event may be valid when it contain whitespace
            line = line[len(b"data: ") :]
        else:
            line = line[len(b"data:") :]
        if line.strip() == b"[DONE]":
            # return here will cause GeneratorExit exception in urllib3
            # and it will close http connection with TCP Reset
            return None
        else:
            return line.decode("utf-8")
    return None


def _handle_sse_line(line: str) -> Optional[BaseMessageChunk]:
    try:
        obj = json.loads(line)
        default_chunk_class = AIMessageChunk
        delta = obj.get("choices", [{}])[0].get("delta", {})
        return _convert_delta_to_message_chunk(delta, default_chunk_class)
    except Exception:
        return None
