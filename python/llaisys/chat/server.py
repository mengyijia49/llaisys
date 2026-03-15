import argparse
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import llaisys

try:
    from pydantic import BaseModel
except Exception as e:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "Pydantic is required for the chat server. Install with: pip install pydantic"
    ) from e

try:
    from fastapi import FastAPI, HTTPException as FastAPIHTTPException
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
except Exception:  # pragma: no cover - optional until build_app() is called
    FastAPI = None
    FileResponse = None
    JSONResponse = None
    StreamingResponse = None

    class HTTPException(RuntimeError):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = int(status_code)
            self.detail = detail
else:  # pragma: no cover - exercised at runtime
    HTTPException = FastAPIHTTPException


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 20
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    stream: bool = False
    seed: Optional[int] = None


def _as_llaisys_device(device_name: str) -> llaisys.DeviceType:
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    if device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    raise ValueError(f"Unsupported device: {device_name}")


def _max_new_tokens(req: ChatCompletionRequest) -> int:
    if req.max_completion_tokens is not None:
        return max(1, int(req.max_completion_tokens))
    if req.max_tokens is not None:
        return max(1, int(req.max_tokens))
    return 128


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


_FINAL_ANSWER_MARKERS = (
    r"(?i)\bfinal answer\b\s*[:：-]\s*",
    r"(?i)\banswer\b\s*[:：-]\s*",
    r"最终答案\s*[:：]\s*",
    r"答案\s*[:：]\s*",
    r"(?i)\bthe answer is\b\s*",
    r"(?i)\bso the answer is\b\s*",
    r"答案是",
)
_LEADING_REASONING_CLAUSES = (
    re.compile(r"^\s*(?:alright|okay|ok|well|sure)[,，:\-\s]+", flags=re.IGNORECASE),
    re.compile(r"^\s*let me (?:try to )?(?:break this down|think(?: this through)?|answer|respond|be direct|explain)[：:,\-\s]*", flags=re.IGNORECASE),
    re.compile(r"^\s*the user (?:asks?|asked|wants|is asking)[^.。!?]*[.。!?]\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*looking at (?:the )?(?:history|conversation)[^.。!?]*[.。!?]\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*based on (?:the )?(?:history|conversation)[^.。!?]*[.。!?]\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*from (?:the )?(?:history|conversation)[^.。!?]*[.。!?]\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*hmm[^.。!?]*[.。!?]\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*maybe[^.。!?]*[.。!?]\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*also[^.。!?]*[.。!?]\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*so,?\s+now[^.。!?]*[.。!?]\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*让我(?:直接)?回答[：:，,\s]*"),
    re.compile(r"^\s*直接回答[：:，,\s]*"),
    re.compile(r"^\s*好的[，,\s]*"),
    re.compile(r"^\s*用户[^。！？!?]*[。！？!?]\s*"),
    re.compile(r"^\s*从(?:对话|历史|上下文)来看[^。！？!?]*[。！？!?]\s*"),
)
_LEAKED_REASONING_PREFIX = re.compile(
    r"^\s*(?:"
    r"alright\b.*|"
    r"okay\b.*|"
    r"ok\b.*|"
    r"hmm\b.*|"
    r"let me\b.*|"
    r"let's\b.*|"
    r"i need to\b.*|"
    r"i should\b.*|"
    r"the user\b.*|"
    r"looking at (?:the )?(?:history|conversation)\b.*|"
    r"based on (?:the )?(?:history|conversation)\b.*|"
    r"from (?:the )?(?:history|conversation)\b.*|"
    r"maybe\b.*|"
    r"also\b.*|"
    r"so,?\s+now\b.*|"
    r"好的[，,]?\s*让我.*|"
    r"让我.*|"
    r"用户.*|"
    r"从(?:对话|历史|上下文)来看.*|"
    r"看起来.*|"
    r"也许.*|"
    r"我(?:需要|应该|先).*"
    r")\s*$",
    flags=re.IGNORECASE,
)
_CLEAN_FINAL_ANSWER_FALLBACK = "抱歉，本轮没有生成可直接展示的最终答案，请重试。"
_MODEL_SPECIAL_TOKEN_RE = re.compile(
    r"<｜(?:begin▁of▁sentence|end▁of▁sentence|User|Assistant|System)｜>"
)


def _trim_reasoning_intro(text: str) -> str:
    text = str(text or "")
    changed = True
    while changed and text:
        changed = False
        for pattern in _LEADING_REASONING_CLAUSES:
            updated = pattern.sub("", text, count=1)
            updated = re.sub(r"^[\s\.,;:，。；：!?！？-]+", "", updated)
            if updated != text:
                text = updated
                changed = True
                break
    text = re.sub(r"^[\s\.,;:，。；：!?！？-]+", "", text)
    return text.strip()


def _strip_leaked_reasoning(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    for marker in _FINAL_ANSWER_MARKERS:
        match = re.search(marker, text)
        if match:
            return text[match.end():].strip()

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    while paragraphs:
        trimmed = _trim_reasoning_intro(paragraphs[0])
        if trimmed and not _LEAKED_REASONING_PREFIX.match(trimmed):
            paragraphs[0] = trimmed
            break
        if trimmed and trimmed != paragraphs[0]:
            paragraphs[0] = trimmed
            continue
        if not _LEAKED_REASONING_PREFIX.match(paragraphs[0]):
            break
        paragraphs.pop(0)
    if paragraphs:
        text = "\n\n".join(paragraphs).strip()
    else:
        text = ""
    return text


def _looks_like_leaked_reasoning(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    first_block = re.split(r"\n\s*\n+", text, maxsplit=1)[0].strip()
    return bool(_LEAKED_REASONING_PREFIX.match(first_block))


def _clean_assistant_text(text: str) -> str:
    # Remove explicit thinking traces often produced by reasoning checkpoints.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    text = _MODEL_SPECIAL_TOKEN_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    text = _strip_leaked_reasoning(text)
    return text.strip()


def _relaxed_assistant_text(text: str) -> str:
    # Fallback cleaner: keep content but drop tags/control characters.
    text = text.replace("<think>", "").replace("</think>", "")
    text = _MODEL_SPECIAL_TOKEN_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return text.strip()


def _strip_forced_think_prefix(prompt: str) -> str:
    think_prefix = "<｜Assistant｜><think>\n"
    if prompt.endswith(think_prefix):
        return prompt[: -len(think_prefix)] + "<｜Assistant｜>"
    return prompt


def _webui_index_path() -> Path:
    return Path(__file__).with_name("webui") / "index.html"


class ChatEngine:
    def __init__(
        self,
        model_path: str,
        device_name: str,
        model_name: str,
        engine: str,
        default_system_prompt: str,
    ):
        from transformers import AutoTokenizer

        self.model_name = model_name
        self.engine = engine
        self.default_system_prompt = default_system_prompt.strip()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = llaisys.models.Qwen2(model_path, _as_llaisys_device(device_name), backend=engine)
        self.lock = threading.Lock()
        self.max_context_tokens = int(getattr(self.model, "_maxseq", 0) or 0)
        self.special_token_ids = {int(t) for t in getattr(self.tokenizer, "all_special_ids", []) or []}
        self.eos_token_ids = set()
        tok_eos = getattr(self.tokenizer, "eos_token_id", None)
        if tok_eos is not None:
            self.eos_token_ids.add(int(tok_eos))
        model_eos = getattr(self.model, "_end_token", -1)
        if isinstance(model_eos, int) and model_eos >= 0:
            self.eos_token_ids.add(int(model_eos))

    def _contains_eos(self, token_ids: List[int]) -> bool:
        if not self.eos_token_ids:
            return False
        for t in token_ids:
            if int(t) in self.eos_token_ids:
                return True
        return False

    def _normalize_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        conv = [{"role": m.role, "content": m.content} for m in messages]
        if self.default_system_prompt and not any(m["role"] == "system" for m in conv):
            conv = [{"role": "system", "content": self.default_system_prompt}] + conv
        return conv

    def _encode_prompt(self, conversation: List[Dict[str, str]]) -> List[int]:
        prompt = self.tokenizer.apply_chat_template(
            conversation=conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt = _strip_forced_think_prefix(prompt)
        return self.tokenizer.encode(prompt)

    def _truncate_latest_message(
        self,
        conversation: List[Dict[str, str]],
        prompt_budget: int,
        trimmed_messages: int,
    ) -> tuple[List[Dict[str, str]], int, bool]:
        if not conversation:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        last_idx = len(conversation) - 1
        last_message = dict(conversation[last_idx])
        content = str(last_message.get("content") or "")
        content_ids = self.tokenizer.encode(content, add_special_tokens=False)
        if not content_ids:
            raise HTTPException(
                status_code=400,
                detail="the latest message is too long for the model context window",
            )

        prefix = "[Earlier content omitted]\n" if last_message.get("role") != "system" else ""
        best = None
        lo, hi = 1, len(content_ids)
        while lo <= hi:
            mid = (lo + hi) // 2
            clipped = prefix + self.tokenizer.decode(
                content_ids[-mid:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            trial = list(conversation)
            trial[last_idx] = {"role": last_message["role"], "content": clipped}
            if len(self._encode_prompt(trial)) <= prompt_budget:
                best = trial
                lo = mid + 1
            else:
                hi = mid - 1

        if best is None:
            raise HTTPException(
                status_code=400,
                detail="the latest message is too long for the model context window",
            )
        return best, trimmed_messages, True

    def _trim_conversation(
        self,
        conversation: List[Dict[str, str]],
        requested_completion_tokens: int,
    ) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
        context_info: Dict[str, Any] = {
            "context_window": self.max_context_tokens or None,
            "requested_completion_tokens": requested_completion_tokens,
            "effective_completion_tokens": requested_completion_tokens,
            "trimmed_messages": 0,
            "latest_message_truncated": False,
        }
        if self.max_context_tokens <= 0:
            return conversation, context_info

        prompt_budget = max(1, self.max_context_tokens - requested_completion_tokens)
        if len(self._encode_prompt(conversation)) <= prompt_budget:
            return conversation, context_info

        system_message = conversation[0] if conversation and conversation[0]["role"] == "system" else None
        body = list(conversation[1:] if system_message is not None else conversation)
        while body:
            candidate = ([system_message] if system_message is not None else []) + body
            if body[0]["role"] != "assistant" and len(self._encode_prompt(candidate)) <= prompt_budget:
                context_info["trimmed_messages"] = len(conversation) - len(candidate)
                return candidate, context_info
            body = body[1:]
            while body and body[0]["role"] == "assistant":
                body = body[1:]

        minimal: List[Dict[str, str]] = []
        if system_message is not None:
            minimal.append(system_message)
        if conversation:
            if (
                len(conversation) >= 2
                and conversation[-1]["role"] == "assistant"
                and conversation[-2]["role"] == "user"
            ):
                minimal.extend(conversation[-2:])
            elif conversation[-1]["role"] != "system":
                minimal.append(conversation[-1])

        if not minimal or minimal[-1]["role"] == "system":
            raise HTTPException(status_code=400, detail="messages must contain at least one non-system message")

        if len(self._encode_prompt(minimal)) <= prompt_budget:
            context_info["trimmed_messages"] = len(conversation) - len(minimal)
            return minimal, context_info

        minimal, trimmed_messages, latest_message_truncated = self._truncate_latest_message(
            minimal,
            prompt_budget=prompt_budget,
            trimmed_messages=len(conversation) - len(minimal),
        )
        context_info["trimmed_messages"] = trimmed_messages
        context_info["latest_message_truncated"] = latest_message_truncated
        return minimal, context_info

    def _prepare_request(self, req: ChatCompletionRequest) -> tuple[List[int], int, Dict[str, Any]]:
        requested_completion_tokens = _max_new_tokens(req)
        conversation = self._normalize_messages(req.messages)
        conversation, context_info = self._trim_conversation(conversation, requested_completion_tokens)
        prompt_ids = self._encode_prompt(conversation)

        effective_completion_tokens = requested_completion_tokens
        if self.max_context_tokens > 0:
            available_completion = self.max_context_tokens - len(prompt_ids)
            if available_completion <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"prompt is too long for the model context window ({self.max_context_tokens} tokens)",
                )
            effective_completion_tokens = min(requested_completion_tokens, available_completion)
            context_info["effective_completion_tokens"] = effective_completion_tokens

        return prompt_ids, effective_completion_tokens, context_info

    def _append_cache_info(self, context_info: Dict[str, Any]) -> Dict[str, Any]:
        getter = getattr(self.model, "get_last_cache_info", None)
        if not callable(getter):
            return context_info
        info = getter() or {}
        if not info:
            return context_info
        if "cache_hit" in info:
            context_info["prefix_cache_hit"] = bool(info.get("cache_hit"))
        reused = int(info.get("reused_prefix_tokens", 0) or 0)
        if reused > 0:
            context_info["reused_prefix_tokens"] = reused
        return context_info

    def _commit_visible_prefix_cache(
        self,
        req: ChatCompletionRequest,
        prompt_ids: List[int],
        visible_text: str,
        max_budget: int,
    ) -> None:
        if not req.session_id or not visible_text:
            return
        committer = getattr(self.model, "commit_prefix_cache", None)
        if not callable(committer):
            return
        visible_ids = self.tokenizer.encode(
            visible_text,
            add_special_tokens=False,
        )
        committer(
            req.session_id,
            prompt_ids,
            visible_ids,
            extra_capacity=max_budget,
        )

    def complete(self, req: ChatCompletionRequest, request_id: str, created: int) -> Dict:
        prompt_ids, max_budget, context_info = self._prepare_request(req)
        completion_ids: List[int] = []
        finish_reason = "stop"

        for token_id in self.model.stream_generate(
            prompt_ids,
            max_new_tokens=max_budget,
            top_k=req.top_k,
            top_p=req.top_p,
            temperature=req.temperature,
            seed=req.seed,
            repetition_penalty=req.repetition_penalty,
            no_repeat_ngram_size=req.no_repeat_ngram_size,
            cache_key=req.session_id,
        ):
            token_id = int(token_id)
            completion_ids.append(token_id)
            if self._contains_eos([token_id]):
                break

        if (
            len(completion_ids) >= max_budget
            and completion_ids
            and not self._contains_eos([completion_ids[-1]])
        ):
            finish_reason = "length"

        raw_content = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        content = _clean_assistant_text(raw_content)
        if not content:
            content = _relaxed_assistant_text(raw_content)
            if _looks_like_leaked_reasoning(content):
                content = ""
        if not content:
            content = _CLEAN_FINAL_ANSWER_FALLBACK
        self._commit_visible_prefix_cache(req, prompt_ids, content, max_budget)
        context_info = self._append_cache_info(context_info)
        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": req.model or self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "total_tokens": len(prompt_ids) + len(completion_ids),
            },
            "llaisys": context_info,
        }

    def stream_complete(self, req: ChatCompletionRequest, request_id: str, created: int) -> Iterator[str]:
        prompt_ids, max_budget, context_info = self._prepare_request(req)
        model_name = req.model or self.model_name
        completion_tokens = 0
        raw_text = ""
        emitted_text = ""
        completion_ids: List[int] = []
        finish_reason = "stop"

        yield _sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
        )

        for token_id in self.model.stream_generate(
            prompt_ids,
            max_new_tokens=max_budget,
            top_k=req.top_k,
            top_p=req.top_p,
            temperature=req.temperature,
            seed=req.seed,
            repetition_penalty=req.repetition_penalty,
            no_repeat_ngram_size=req.no_repeat_ngram_size,
            cache_key=req.session_id,
        ):
            token_id = int(token_id)
            completion_ids.append(token_id)
            completion_tokens += 1
            if token_id not in self.special_token_ids:
                text_piece = self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if text_piece:
                    raw_text += text_piece
                    cleaned = _clean_assistant_text(raw_text)
                    if cleaned:
                        if cleaned.startswith(emitted_text):
                            text_piece = cleaned[len(emitted_text):]
                        else:
                            text_piece = cleaned
                        if text_piece:
                            emitted_text = cleaned
                            yield _sse(
                                {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_name,
                                    "choices": [{"index": 0, "delta": {"content": text_piece}, "finish_reason": None}],
                                }
                            )

            if self._contains_eos([token_id]):
                break

        if (
            len(completion_ids) >= max_budget
            and completion_ids
            and not self._contains_eos([completion_ids[-1]])
        ):
            finish_reason = "length"

        if not emitted_text:
            fallback_text = _relaxed_assistant_text(raw_text)
            if _looks_like_leaked_reasoning(fallback_text):
                fallback_text = ""
            if not fallback_text:
                fallback_text = _CLEAN_FINAL_ANSWER_FALLBACK
            yield _sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": fallback_text}, "finish_reason": None}],
                    }
                )
            emitted_text = fallback_text

        self._commit_visible_prefix_cache(req, prompt_ids, emitted_text, max_budget)
        context_info = self._append_cache_info(context_info)

        yield _sse(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": completion_tokens,
                    "total_tokens": len(prompt_ids) + completion_tokens,
                },
                "llaisys": context_info,
            }
        )
        yield "data: [DONE]\n\n"


def build_app(
    model_path: str,
    device_name: str = "cpu",
    model_name: str = "llaisys-qwen2",
    engine: str = "llaisys",
    default_system_prompt: str = "",
) -> FastAPI:
    if FastAPI is None or FileResponse is None or JSONResponse is None or StreamingResponse is None:
        raise RuntimeError("FastAPI dependencies are missing. Install with: pip install fastapi uvicorn")

    app = FastAPI(title="LLAISYS Chat API")
    chat_engine = ChatEngine(
        model_path=model_path,
        device_name=device_name,
        model_name=model_name,
        engine=engine,
        default_system_prompt=default_system_prompt,
    )

    @app.get("/")
    def root():
        return {
            "name": "LLAISYS Chat API",
            "health": "/health",
            "chat_completions": "/v1/chat/completions",
            "chat_ui": "/chat",
            "engine": engine,
            "default_system_prompt_enabled": bool(default_system_prompt),
        }

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "engine": engine}

    @app.get("/chat", include_in_schema=False)
    @app.get("/chat/", include_in_schema=False)
    def chat_ui():
        index_path = _webui_index_path()
        if not index_path.is_file():
            raise HTTPException(status_code=500, detail="chat web UI is missing from the package")
        return FileResponse(index_path)

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        if len(req.messages) == 0:
            raise HTTPException(status_code=400, detail="messages must not be empty")

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        if req.stream:
            def stream():
                with chat_engine.lock:
                    yield from chat_engine.stream_complete(req, request_id, created)

            return StreamingResponse(stream(), media_type="text/event-stream")

        with chat_engine.lock:
            resp = chat_engine.complete(req, request_id, created)
        return JSONResponse(resp)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="Path to model directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--engine", default="llaisys", choices=["auto", "llaisys", "hf"], type=str)
    parser.add_argument(
        "--system-prompt",
        default=(
            "You are a concise assistant. Reply in the same language as the user. "
            "Give only the final answer, ideally in one short paragraph or one sentence for simple factual questions. "
            "Do not reveal thinking steps. Do not analyze the conversation history. "
            "Do not repeat or paraphrase the user's question unless the user explicitly asks you to."
        ),
        type=str,
    )
    parser.add_argument("--model-name", default="llaisys-qwen2", type=str)
    parser.add_argument("--host", default="127.0.0.1", type=str)
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    import uvicorn

    app = build_app(
        args.model,
        device_name=args.device,
        model_name=args.model_name,
        engine=args.engine,
        default_system_prompt=args.system_prompt,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
