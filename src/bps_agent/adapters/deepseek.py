"""OpenAI Chat-compatible DeepSeek adjudication adapter."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from bps_agent.models import EvidenceBundle, ProviderConfig, VerdictDocument

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
_SYSTEM_PROMPT = """你是网络设备性能测试裁决专家。
请只依据给定 Evidence Bundle 判断本次性能测试是否通过。
你拥有测试 Verdict 的最终裁决权。
证据充分且达到当前流量目标时返回 pass。
未达到当前流量目标时返回 retry，系统会按预定降载策略开始下一次测试。
必须返回一个 JSON 对象, 顶层 verdict 只能是 pass 或 retry。
可以加入 summary、observations、risks、retry_reason、confidence 等字段。
不得输出 JSON 之外的文字, 不得臆造证据中不存在的指标。
请把 bps_performance_analysis 作为确定性性能波动证据纳入判断，但它不替代你的最终 Verdict。
不要把基础设施错误当作 DUT 性能失败。"""


class ProviderCompatibilityError(RuntimeError):
    pass


class DeepSeekJudge:
    def __init__(
        self,
        provider_name: str,
        config: ProviderConfig,
        *,
        token: str,
        client: httpx.Client | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = config.model
        self.config = config
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds),
            follow_redirects=False,
            trust_env=False,
        )
        self._owns_client = client is None
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    @property
    def _endpoint(self) -> str:
        return f"{self.config.base_url}/chat/completions"

    def _request_verdict(
        self, messages: list[dict[str, str]]
    ) -> tuple[VerdictDocument, dict[str, Any]]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled"},
            "reasoning_effort": "max",
        }
        last_error: Exception | None = None
        for attempt in range(self.config.attempts):
            try:
                response = self._client.post(self._endpoint, headers=self._headers, json=payload)
                if response.is_redirect:
                    raise RuntimeError("LLM endpoint returned an unexpected redirect")
                if response.status_code in _RETRYABLE_STATUS and attempt + 1 < self.config.attempts:
                    time.sleep(min(2**attempt, 10))
                    continue
                if response.status_code == 400:
                    raise ProviderCompatibilityError(
                        "LLM rejected JSON/thinking/reasoning_effort=max compatibility: "
                        f"{response.text[:500]}"
                    )
                response.raise_for_status()
                document = response.json()
                if not isinstance(document, dict):
                    raise RuntimeError("LLM response envelope is not a JSON object")
                parsed = json.loads(self._content(document))
                return VerdictDocument.model_validate(parsed), {
                    "request": payload,
                    "response": document,
                }
            except ProviderCompatibilityError:
                raise
            except (
                httpx.NetworkError,
                httpx.TimeoutException,
                json.JSONDecodeError,
                RuntimeError,
                ValueError,
            ) as exc:
                last_error = exc
                if attempt + 1 >= self.config.attempts:
                    break
                time.sleep(min(2**attempt, 10))
        raise RuntimeError(f"LLM did not return a valid Verdict after retries: {last_error}")

    @staticmethod
    def _content(document: dict[str, Any]) -> str:
        choices = document.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError("LLM response omitted choices")
        message = choices[0].get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("LLM response omitted message content")
        content = message["content"]
        assert isinstance(content, str)
        return content

    def validate_compatibility(self) -> None:
        self._request_verdict(
            [
                {"role": "system", "content": '只返回 JSON：{"verdict":"pass"}'},
                {"role": "user", "content": "这是接口兼容性检查，不是实际测试裁决。"},
            ]
        )

    def adjudicate(self, evidence: EvidenceBundle) -> tuple[VerdictDocument, dict[str, Any]]:
        user_content = json.dumps(evidence.model_dump(mode="json"), ensure_ascii=False)
        return self._request_verdict(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )

    def close(self) -> None:
        self._headers["Authorization"] = "Bearer <cleared>"
        if self._owns_client:
            self._client.close()
