"""OpenAI-compatible API client with retry, concurrency control, and usage logging."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import fcntl
except ImportError:  # Optional cross-process rate limiting requires POSIX.
    fcntl = None  # type: ignore[assignment]

try:
    import httpx
except ImportError:  # pragma: no cover - surfaced as a runtime configuration error
    httpx = None  # type: ignore[assignment]

try:
    import openai
except ImportError:  # pragma: no cover - surfaced as a runtime configuration error
    openai = None  # type: ignore[assignment]

_LOG_DIR = Path(os.environ.get("POINTQ_LOG_DIR") or "outputs/logs")
_DEFAULT_RATE_LIMIT_WINDOW_SEC = 60.0


def _emit_rate_limit_notice(message: str) -> None:
    text = str(message).rstrip() + "\n"
    try:
        with open("/dev/tty", "w", encoding="utf-8", buffering=1) as tty:
            tty.write(text)
            tty.flush()
            return
    except OSError:
        pass

    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except Exception:
        pass


def _safe_get(obj: Any, path: List[str], default=None):
    cur = obj
    for k in path:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            try:
                cur = getattr(cur, k)
            except Exception:
                return default
    return cur if cur is not None else default


def extract_usage(resp: Any) -> Dict[str, Optional[int]]:
    usage = _safe_get(resp, ["usage"], {}) or {}
    # Web search / tool billing: shape varies by gateway (OpenAI vs proxies).
    web_search = (
        _safe_int(_safe_get(usage, ["web_search_requests"]))
        or _safe_int(_safe_get(usage, ["server_tool_usage", "web_search_requests"]))
        or _safe_int(_safe_get(usage, ["input_tokens_details", "web_search_requests"]))
    )
    return {
        "input_tokens": _safe_int(_safe_get(usage, ["prompt_tokens"])),
        "output_tokens": _safe_int(_safe_get(usage, ["completion_tokens"])),
        "total_tokens": _safe_int(_safe_get(usage, ["total_tokens"])),
        "cached_tokens": _safe_int(_safe_get(usage, ["prompt_tokens_details", "cached_tokens"])),
        "reasoning_tokens": _safe_int(_safe_get(usage, ["completion_tokens_details", "reasoning_tokens"])),
        "web_search_requests": web_search,
    }


def _safe_int(v):
    return int(v) if v is not None else None


def _extract_first_choice(resp: Any) -> Any:
    choices = _safe_get(resp, ["choices"], []) or []
    if isinstance(choices, list) and choices:
        return choices[0]
    return None


def _coerce_text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    chunks.append(text)
                    continue
            text = _safe_get(item, ["text"])
            if isinstance(text, str):
                chunks.append(text)
        return "".join(chunks)
    return str(content)


def _preview_value(value: Any, *, max_chars: int = 200) -> Optional[str]:
    if value is None:
        return None
    text = _coerce_text_content(value) if isinstance(value, (str, list)) else repr(value)
    if not text:
        return ""
    text = text.replace("\n", "\\n")
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return text


def extract_response_details(
    resp: Any,
    *,
    requested_max_completion_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    choice = _extract_first_choice(resp)
    message = _safe_get(choice, ["message"], {}) or {}
    raw_content = _safe_get(message, ["content"])
    text = _coerce_text_content(raw_content)
    usage = extract_usage(resp)
    return {
        "text": text,
        "usage": usage,
        "finish_reason": _safe_get(choice, ["finish_reason"]),
        "requested_max_completion_tokens": _safe_int(requested_max_completion_tokens),
        "response_id": _safe_get(resp, ["id"]),
        "response_model": _safe_get(resp, ["model"]),
        "raw_content_preview": _preview_value(raw_content),
        "raw_content_type": None if raw_content is None else type(raw_content).__name__,
        "raw_content_is_none": raw_content is None,
        "content_was_empty": text.strip() == "",
    }


def _extract_http_status(exc: BaseException) -> Optional[int]:
    direct = _safe_int(getattr(exc, "status_code", None))
    if direct is not None:
        return direct

    response = getattr(exc, "response", None)
    if response is not None:
        status = _safe_int(getattr(response, "status_code", None))
        if status is not None:
            return status
        status = _safe_int(_safe_get(response, ["status_code"]))
        if status is not None:
            return status

    text = str(exc or "")
    for pattern in (
        r"Error code:\s*(\d{3})",
        r"['\"]status['\"]\s*:\s*(\d{3})",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def describe_api_error(exc: BaseException) -> Dict[str, Any]:
    message = str(exc or "").strip()
    lowered = message.lower()
    status = _extract_http_status(exc)
    exc_type = exc.__class__.__name__
    exc_type_lower = exc_type.lower()

    timeout_types = ()
    connection_types = ()
    if httpx is not None:
        timeout_types = (httpx.TimeoutException,)
        connection_types = (
            httpx.ConnectError,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
            httpx.ProtocolError,
            httpx.NetworkError,
        )

    if (
        status == 401
        or "invalid_api_key" in lowered
        or "incorrect api key" in lowered
        or "unauthorized" in lowered
        or "authentication" in lowered
    ):
        error_kind = "auth_error"
    elif status == 429 or "rate limit" in lowered or "too many requests" in lowered:
        error_kind = "rate_limit_error"
    elif (
        (timeout_types and isinstance(exc, timeout_types))
        or "timed out" in lowered
        or "timeout" in lowered
    ):
        error_kind = "timeout_error"
    elif (
        (connection_types and isinstance(exc, connection_types))
        or "connection error" in lowered
        or "connectionerror" in exc_type_lower
        or "connecterror" in exc_type_lower
        or "apiconnectionerror" in exc_type_lower
    ):
        error_kind = "connection_error"
    else:
        error_kind = "unknown_error"

    return {
        "error_kind": error_kind,
        "http_status": status,
        "message": message,
        "exception_type": exc_type,
    }


def append_usage_log(record: Dict[str, Any]):
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOG_DIR / "benchmark_usage_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_usage_record(
    *,
    run_id: str,
    model: str,
    task_name: str,
    sample_id: str,
    view_setting: str,
    run_idx: int,
    usage: Dict[str, Optional[int]],
    latency_sec: float,
    success: bool,
    error_msg: Optional[str] = None,
    error_kind: Optional[str] = None,
    http_status: Optional[int] = None,
    response_preview: Optional[str] = None,
    estimated_cost: Optional[float] = None,
    finish_reason: Optional[str] = None,
    requested_max_completion_tokens: Optional[int] = None,
    raw_content_preview: Optional[str] = None,
    content_was_empty: Optional[bool] = None,
    truncation_suspected: Optional[bool] = None,
) -> Dict[str, Any]:
    rec = {
        "ts": datetime.now().isoformat(),
        "run_id": run_id,
        "model": model,
        "task_name": task_name,
        "sample_id": sample_id,
        "view_setting": view_setting,
        "run_idx": run_idx,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": usage.get("cached_tokens"),
        "web_search_requests": usage.get("web_search_requests"),
        "latency_sec": round(latency_sec, 3),
        "success": success,
        "error_msg": error_msg,
        "error_kind": error_kind,
        "http_status": http_status,
        "response_preview": (response_preview or "")[:300],
    }
    if estimated_cost is not None:
        rec["estimated_cost_usd"] = round(estimated_cost, 6)
    if finish_reason is not None:
        rec["finish_reason"] = str(finish_reason)
    if requested_max_completion_tokens is not None:
        rec["requested_max_completion_tokens"] = int(requested_max_completion_tokens)
    if raw_content_preview is not None:
        rec["raw_content_preview"] = str(raw_content_preview)[:300]
    if content_was_empty is not None:
        rec["content_was_empty"] = bool(content_was_empty)
    if truncation_suspected is not None:
        rec["truncation_suspected"] = bool(truncation_suspected)
    return rec


class VLMClient:
    """Async-capable OpenAI-compatible VLM client with retry and concurrency."""

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.openai.com/v1",
        max_concurrent: int = 8,
        max_retries: int = 1,
        retry_backoff: float = 1.0,
        timeout: float = 120.0,
        dry_run: bool = False,
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._timeout = timeout
        self.dry_run = dry_run

    @staticmethod
    def _rate_limit_config(model: str, api_base: str) -> Optional[Dict[str, Any]]:
        raw_limit = str(os.environ.get("VLM_RATE_LIMIT_RPM") or "").strip()
        if not raw_limit:
            return None
        try:
            limit = int(raw_limit)
        except ValueError:
            return None
        if limit <= 0:
            return None

        raw_window = str(os.environ.get("VLM_RATE_LIMIT_WINDOW_SEC") or "").strip()
        try:
            window_sec = float(raw_window) if raw_window else _DEFAULT_RATE_LIMIT_WINDOW_SEC
        except ValueError:
            window_sec = _DEFAULT_RATE_LIMIT_WINDOW_SEC
        if window_sec <= 0:
            window_sec = _DEFAULT_RATE_LIMIT_WINDOW_SEC

        scope = str(os.environ.get("VLM_RATE_LIMIT_SCOPE") or "").strip()
        if not scope:
            scope = f"{api_base}|{model}"
        return {
            "limit": limit,
            "window_sec": window_sec,
            "scope": scope,
        }

    @staticmethod
    def _rate_limit_state_path(scope: str) -> Path:
        base_dir = Path(os.environ.get("VLM_RATE_LIMIT_DIR") or (Path(tempfile.gettempdir()) / "pointq-vlm-rate-limit"))
        base_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(scope.encode("utf-8")).hexdigest()
        return base_dir / f"{digest}.json"

    def _acquire_rate_limit_slot_sync(self, model: str) -> None:
        config = self._rate_limit_config(model, self.api_base)
        if config is None:
            return
        if fcntl is None:
            raise RuntimeError("fcntl is required for cross-process rate limiting on this platform.")

        limit = int(config["limit"])
        window_sec = float(config["window_sec"])
        state_path = self._rate_limit_state_path(str(config["scope"]))

        while True:
            wait_sec = 0.0
            with open(state_path, "a+", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.seek(0)
                    raw = handle.read().strip()
                    loaded = json.loads(raw) if raw else []
                    if not isinstance(loaded, list):
                        loaded = []

                    now = time.time()
                    recent: List[float] = []
                    for item in loaded:
                        if isinstance(item, (int, float)):
                            ts = float(item)
                            if ts > now - window_sec:
                                recent.append(ts)
                    recent.sort()

                    if len(recent) < limit:
                        recent.append(now)
                        handle.seek(0)
                        handle.truncate()
                        json.dump(recent, handle)
                        handle.flush()
                        return

                    wait_sec = max(0.05, recent[0] + window_sec - now)
                    resume_at = datetime.fromtimestamp(now + wait_sec).strftime("%H:%M:%S")
                    rate_limit_label = str(os.environ.get("VLM_RATE_LIMIT_LABEL") or "").strip()
                    label_text = f"[{rate_limit_label}]" if rate_limit_label else ""
                    _emit_rate_limit_notice(
                        f"[RATE-LIMIT][model={model}][pid={os.getpid()}]{label_text} "
                        f"限流中，{wait_sec:.1f} 秒后继续（预计 {resume_at}）"
                    )
                    handle.seek(0)
                    handle.truncate()
                    json.dump(recent, handle)
                    handle.flush()
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

            time.sleep(wait_sec)

    async def _acquire_rate_limit_slot_async(self, model: str) -> None:
        await asyncio.to_thread(self._acquire_rate_limit_slot_sync, model)

    def _ensure_dependencies(self) -> None:
        if openai is None or httpx is None:
            missing = []
            if openai is None:
                missing.append("openai")
            if httpx is None:
                missing.append("httpx")
            joined = ", ".join(missing)
            raise RuntimeError(
                f"Missing required dependency: {joined}. "
                "Install them first, e.g. `pip install openai httpx`."
            )

    @staticmethod
    def _build_request_kwargs(
        *,
        temperature: float,
        reasoning_effort: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"temperature": temperature}
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = int(max_completion_tokens)
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
        if extra_body:
            kwargs["extra_body"] = extra_body
        return kwargs

    def _build_sync_client(self) -> openai.OpenAI:
        self._ensure_dependencies()
        return openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=httpx.Client(timeout=self._timeout),
        )

    def _build_async_client(self) -> openai.AsyncOpenAI:
        self._ensure_dependencies()
        return openai.AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            http_client=httpx.AsyncClient(timeout=self._timeout),
        )

    _DRY_USAGE: Dict[str, Optional[int]] = {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
        "cached_tokens": None,
        "reasoning_tokens": None,
        "web_search_requests": None,
    }

    _DRY_RESPONSE_DETAILS: Dict[str, Any] = {
        "text": "[no-api-smoke]",
        "usage": dict(_DRY_USAGE),
        "finish_reason": "stop",
        "requested_max_completion_tokens": None,
        "response_id": "__dry_run__",
        "response_model": "__dry_run__",
        "raw_content_preview": "[no-api-smoke]",
        "raw_content_type": "str",
        "raw_content_is_none": False,
        "content_was_empty": False,
    }

    def call_sync_detailed(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], float]:
        """Synchronous call with response diagnostics. Returns (details_dict, latency)."""
        if self.dry_run:
            details = dict(self._DRY_RESPONSE_DETAILS)
            details["usage"] = dict(self._DRY_USAGE)
            details["requested_max_completion_tokens"] = _safe_int(max_completion_tokens)
            return details, 0.0
        client = self._build_sync_client()
        try:
            last_err = None
            for attempt in range(1, self.max_retries + 1):
                self._acquire_rate_limit_slot_sync(model)
                t0 = time.time()
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        **self._build_request_kwargs(
                            temperature=temperature,
                            reasoning_effort=reasoning_effort,
                            max_completion_tokens=max_completion_tokens,
                            extra_body=extra_body,
                        ),
                    )
                    details = extract_response_details(
                        resp,
                        requested_max_completion_tokens=max_completion_tokens,
                    )
                    return details, time.time() - t0
                except Exception as e:
                    last_err = e
                    if attempt < self.max_retries:
                        time.sleep(self.retry_backoff ** attempt)
            raise last_err  # type: ignore[misc]
        finally:
            client.close()

    def call_sync(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Dict[str, Optional[int]], float]:
        """Synchronous call with retry. Returns (text, usage_dict, latency)."""
        details, latency = self.call_sync_detailed(
            model=model,
            messages=messages,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            extra_body=extra_body,
        )
        return str(details.get("text") or ""), dict(details.get("usage") or {}), latency

    async def call_async_detailed(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> tuple[Dict[str, Any], float]:
        """Async call with semaphore + response diagnostics. Returns (details_dict, latency)."""
        if self.dry_run:
            async with self._semaphore:
                details = dict(self._DRY_RESPONSE_DETAILS)
                details["usage"] = dict(self._DRY_USAGE)
                details["requested_max_completion_tokens"] = _safe_int(max_completion_tokens)
                return details, 0.0
        async with self._semaphore:
            client = self._build_async_client()
            try:
                last_err = None
                for attempt in range(1, self.max_retries + 1):
                    await self._acquire_rate_limit_slot_async(model)
                    t0 = time.time()
                    try:
                        resp = await client.chat.completions.create(
                            model=model,
                            messages=messages,
                            **self._build_request_kwargs(
                                temperature=temperature,
                                reasoning_effort=reasoning_effort,
                                max_completion_tokens=max_completion_tokens,
                                extra_body=extra_body,
                            ),
                        )
                        details = extract_response_details(
                            resp,
                            requested_max_completion_tokens=max_completion_tokens,
                        )
                        return details, time.time() - t0
                    except Exception as e:
                        last_err = e
                        if attempt < self.max_retries:
                            await asyncio.sleep(self.retry_backoff ** attempt)
                raise last_err  # type: ignore[misc]
            finally:
                await client.close()

    async def call_async(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        reasoning_effort: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Dict[str, Optional[int]], float]:
        """Async call with semaphore + retry. Returns (text, usage_dict, latency)."""
        details, latency = await self.call_async_detailed(
            model=model,
            messages=messages,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            extra_body=extra_body,
        )
        return str(details.get("text") or ""), dict(details.get("usage") or {}), latency
