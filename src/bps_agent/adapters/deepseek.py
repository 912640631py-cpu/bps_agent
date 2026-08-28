"""OpenAI Chat-compatible DeepSeek adjudication adapter."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from bps_agent.models import EvidenceBundle, ProviderConfig, ReasoningEffort, VerdictDocument

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
evaluation_mode 为 bps_only 时没有 DUT 证据是正常情况，不得因此判定失败或要求 DUT 指标。
backend_ssh 的 DUT 证据以 metrics_csv 提供连续采样；结合成功/失败采样计数判断证据可靠性。
不要把基础设施错误当作 DUT 性能失败。"""


class ProviderError(RuntimeError):
    pass


class ProviderCompatibilityError(ProviderError):
    pass


class ProviderRequestError(ProviderError):
    pass


class ProviderResponseError(ProviderError):
    pass


class DeepSeekJudge:
    def __init__(
        self,
        provider_name: str,
        config: ProviderConfig,
        *,
        token: str,
        reasoning_effort: ReasoningEffort = "max",
        client: httpx.Client | None = None,
    ) -> None:
        self.provider_name = provider_name
        self.model_name = config.model
        self.config = config
        self.reasoning_effort = reasoning_effort
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
            "reasoning_effort": self.reasoning_effort,
        }
        last_error: RuntimeError | None = None
        last_cause: Exception | None = None
        for attempt in range(self.config.attempts):
            try:
                response = self._client.post(self._endpoint, headers=self._headers, json=payload)
            except httpx.HTTPError as exc:
                last_error = ProviderRequestError(f"LLM request transport failed: {exc}")
                last_cause = exc
            else:
                if response.is_redirect:
                    raise ProviderRequestError("LLM endpoint returned an unexpected redirect")
                if response.status_code == 400:
                    raise ProviderCompatibilityError(
                        "LLM rejected JSON/thinking/reasoning_effort="
                        f"{self.reasoning_effort} compatibility: "
                        f"{response.text[:500]}"
                    )
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    last_error = ProviderRequestError(
                        f"LLM request failed with HTTP {response.status_code}"
                    )
                    last_cause = exc
                    if response.status_code not in _RETRYABLE_STATUS:
                        raise last_error from exc
                else:
                    try:
                        document = response.json()
                        if not isinstance(document, dict):
                            raise ValueError("LLM response envelope is not a JSON object")
                        parsed = json.loads(self._content(document))
                        verdict = VerdictDocument.model_validate(parsed)
                    except (json.JSONDecodeError, RuntimeError, ValueError) as exc:
                        last_error = ProviderResponseError(f"LLM response was invalid: {exc}")
                        last_cause = exc
                    else:
                        return verdict, {"response": document}
            if attempt + 1 < self.config.attempts:
                time.sleep(min(2**attempt, 10))
        assert last_error is not None
        raise last_error from last_cause

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
        user_content = json.dumps(evidence.as_document(), ensure_ascii=False)
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
