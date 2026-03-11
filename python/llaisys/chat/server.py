import argparse
import json
import re
import threading
import time
import uuid
from typing import Dict, Iterator, List, Optional

import llaisys

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
except Exception as e:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "FastAPI dependencies are missing. Install with: pip install fastapi uvicorn"
    ) from e


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
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


def _sse(payload: Dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _clean_assistant_text(text: str) -> str:
    # Remove explicit thinking traces often produced by reasoning checkpoints.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return text.strip()


def _relaxed_assistant_text(text: str) -> str:
    # Fallback cleaner: keep content but drop tags/control characters.
    text = text.replace("<think>", "").replace("</think>", "")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", text)
    return text.strip()


def _looks_like_sentence_end(text: str) -> bool:
    text = text.rstrip()
    if not text:
        return False
    return text[-1] in "。！？!?…\n”’』】）)]>"


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

    def _max_completion_budget(self, prompt_len: int, step_tokens: int) -> int:
        # Prevent unbounded generation while allowing automatic continuation.
        maxseq = int(getattr(self.model, "_maxseq", 0) or 0)
        if maxseq > 0:
            return max(1, maxseq - prompt_len)
        return max(step_tokens, step_tokens * 8)

    def _build_prompt_ids(self, messages: List[ChatMessage]) -> List[int]:
        conv = [{"role": m.role, "content": m.content} for m in messages]
        if self.default_system_prompt and not any(m["role"] == "system" for m in conv):
            conv = [{"role": "system", "content": self.default_system_prompt}] + conv
        prompt = self.tokenizer.apply_chat_template(
            conversation=conv,
            add_generation_prompt=True,
            tokenize=False,
        )
        # DeepSeek-R1 Distill chat templates often append "<｜Assistant｜><think>\n".
        # Strip the forced think prefix so generation starts from final-answer mode.
        think_prefix = "<｜Assistant｜><think>\n"
        if prompt.endswith(think_prefix):
            prompt = prompt[: -len(think_prefix)] + "<｜Assistant｜>"
        return self.tokenizer.encode(prompt)

    def complete(self, req: ChatCompletionRequest, request_id: str, created: int) -> Dict:
        prompt_ids = self._build_prompt_ids(req.messages)
        step_tokens = _max_new_tokens(req)
        max_budget = self._max_completion_budget(len(prompt_ids), step_tokens)
        completion_ids: List[int] = []
        finish_reason = "stop"

        while len(completion_ids) < max_budget:
            step = min(step_tokens, max_budget - len(completion_ids))
            if step <= 0:
                break
            curr_input = prompt_ids + completion_ids
            step_ids = [
                int(t)
                for t in self.model.stream_generate(
                    curr_input,
                    max_new_tokens=step,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    temperature=req.temperature,
                    seed=req.seed,
                    repetition_penalty=req.repetition_penalty,
                    no_repeat_ngram_size=req.no_repeat_ngram_size,
                )
            ]
            if not step_ids:
                break
            completion_ids.extend(step_ids)
            if self._contains_eos(step_ids):
                break
            if len(step_ids) < step:
                break

            # If this step ended naturally at a sentence boundary, stop.
            tail_ids = completion_ids[-64:]
            tail_text = self.tokenizer.decode(
                tail_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if _looks_like_sentence_end(_clean_assistant_text(tail_text)):
                break

        if len(completion_ids) >= max_budget and (not completion_ids or not self._contains_eos([completion_ids[-1]])):
            finish_reason = "length"

        raw_content = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        content = _clean_assistant_text(raw_content)
        if not content:
            content = _relaxed_assistant_text(raw_content)
        if not content:
            content = "（本轮生成了不可显示的特殊标记，请输入 /regen 重试）"
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
        }

    def stream_complete(self, req: ChatCompletionRequest, request_id: str, created: int) -> Iterator[str]:
        prompt_ids = self._build_prompt_ids(req.messages)
        model_name = req.model or self.model_name
        completion_tokens = 0
        raw_text = ""
        emitted_text = ""
        completion_ids: List[int] = []
        step_tokens = _max_new_tokens(req)
        max_budget = self._max_completion_budget(len(prompt_ids), step_tokens)
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

        while len(completion_ids) < max_budget:
            step = min(step_tokens, max_budget - len(completion_ids))
            if step <= 0:
                break
            curr_input = prompt_ids + completion_ids
            step_ids: List[int] = []

            for token_id in self.model.stream_generate(
                curr_input,
                max_new_tokens=step,
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature,
                seed=req.seed,
                repetition_penalty=req.repetition_penalty,
                no_repeat_ngram_size=req.no_repeat_ngram_size,
            ):
                token_id = int(token_id)
                step_ids.append(token_id)
                completion_ids.append(token_id)
                completion_tokens += 1
                if token_id in self.tokenizer.all_special_ids:
                    continue
                text_piece = self.tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if not text_piece:
                    continue
                raw_text += text_piece
                cleaned = _clean_assistant_text(raw_text)
                if not cleaned:
                    continue
                if cleaned.startswith(emitted_text):
                    text_piece = cleaned[len(emitted_text):]
                else:
                    text_piece = cleaned
                if not text_piece:
                    continue
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

            if not step_ids:
                break
            if self._contains_eos(step_ids):
                break
            if len(step_ids) < step:
                break

            tail_ids = completion_ids[-64:]
            tail_text = self.tokenizer.decode(
                tail_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if _looks_like_sentence_end(_clean_assistant_text(tail_text)):
                break

        if len(completion_ids) >= max_budget and (not completion_ids or not self._contains_eos([completion_ids[-1]])):
            finish_reason = "length"

        if not emitted_text:
            fallback_text = _relaxed_assistant_text(raw_text)
            if not fallback_text:
                fallback_text = "（本轮生成了不可显示的特殊标记，请输入 /regen 重试）"
            yield _sse(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": fallback_text}, "finish_reason": None}],
                }
            )

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
            "engine": engine,
            "default_system_prompt_enabled": bool(default_system_prompt),
        }

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_name, "engine": engine}

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
            "Give only the final answer. Do not reveal thinking steps. "
            "Do not repeat or paraphrase the user's question."
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
