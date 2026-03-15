import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
HELP_TEXT = (
    "Commands: /help /new [name] /switch <name> /list /delete <name> /clear "
    "/history [n] /regen [turn] /edit <turn> <text> /exit"
)


@dataclass
class ChatResult:
    text: str
    meta: Dict


@dataclass
class SessionState:
    messages: List[Dict[str, str]] = field(default_factory=list)


def _sanitize_console_text(text: str) -> str:
    text = ANSI_ESCAPE_RE.sub("", text)
    text = CONTROL_CHAR_RE.sub("", text)
    return text


def _preview(text: str, width: int = 88) -> str:
    text = _sanitize_console_text(text).replace("\n", " / ").strip()
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def _post_json(url: str, payload: Dict):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    return urllib.request.urlopen(req)


def _chat_once(server: str, payload: Dict, stream: bool) -> ChatResult:
    endpoint = f"{server.rstrip('/')}/v1/chat/completions"
    payload = dict(payload)
    payload["stream"] = stream

    if not stream:
        with _post_json(endpoint, payload) as resp:
            body = resp.read().decode("utf-8")
        obj = json.loads(body)
        return ChatResult(
            text=_sanitize_console_text(obj["choices"][0]["message"]["content"]),
            meta=obj.get("llaisys", {}),
        )

    chunks: List[str] = []
    meta: Dict = {}
    with _post_json(endpoint, payload) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            evt = json.loads(data)
            if "llaisys" in evt:
                meta = evt.get("llaisys", {}) or {}
            delta = evt["choices"][0].get("delta", {})
            piece = _sanitize_console_text(delta.get("content", ""))
            if piece:
                print(piece, end="", flush=True)
                chunks.append(piece)
    print()
    return ChatResult(text="".join(chunks), meta=meta)


def _new_session(system_prompt: str) -> SessionState:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    return SessionState(messages=messages)


def _user_turns(messages: List[Dict[str, str]]) -> List[Tuple[int, int, Optional[int]]]:
    turns: List[Tuple[int, int, Optional[int]]] = []
    turn_no = 0
    idx = 0
    while idx < len(messages):
        role = messages[idx]["role"]
        if role == "system":
            idx += 1
            continue
        if role != "user":
            idx += 1
            continue
        turn_no += 1
        assistant_idx = idx + 1 if idx + 1 < len(messages) and messages[idx + 1]["role"] == "assistant" else None
        turns.append((turn_no, idx, assistant_idx))
        idx = idx + 2 if assistant_idx is not None else idx + 1
    return turns


def _resolve_turn(messages: List[Dict[str, str]], ref: str) -> Optional[Tuple[int, int, Optional[int]]]:
    turns = _user_turns(messages)
    if not turns:
        return None
    ref = ref.strip().lower()
    if not ref or ref == "last":
        return turns[-1]
    try:
        wanted = int(ref)
    except ValueError:
        return None
    for turn in turns:
        if turn[0] == wanted:
            return turn
    return None


def _print_history(messages: List[Dict[str, str]], limit: Optional[int] = None):
    if not messages:
        print("(empty session)")
        return

    turns = _user_turns(messages)
    shown_turns = turns[-limit:] if limit is not None and limit > 0 else turns
    shown_user_indices = {user_idx for _, user_idx, _ in shown_turns}

    for idx, msg in enumerate(messages):
        if msg["role"] == "system":
            print(f"system: {_preview(msg['content'])}")
            continue
        if idx not in shown_user_indices and not any(assistant_idx == idx for _, _, assistant_idx in shown_turns):
            continue
        if msg["role"] == "user":
            turn = next(turn_no for turn_no, user_idx, _ in shown_turns if user_idx == idx)
            print(f"turn {turn} user: {_preview(msg['content'])}")
        elif msg["role"] == "assistant":
            turn = next(turn_no for turn_no, _, assistant_idx in shown_turns if assistant_idx == idx)
            print(f"turn {turn} assistant: {_preview(msg['content'])}")
        else:
            print(f"{msg['role']}: {_preview(msg['content'])}")


def _print_server_notes(meta: Dict):
    notes: List[str] = []
    trimmed = int(meta.get("trimmed_messages", 0) or 0)
    if trimmed > 0:
        notes.append(f"trimmed {trimmed} earlier messages to fit context")
    if meta.get("latest_message_truncated"):
        notes.append("truncated the latest message to fit context")
    requested = meta.get("requested_completion_tokens")
    effective = meta.get("effective_completion_tokens")
    if isinstance(requested, int) and isinstance(effective, int) and effective < requested:
        notes.append(f"reduced max completion from {requested} to {effective}")
    if notes:
        print("[context]", "; ".join(notes))


def _request_assistant(args, messages: List[Dict[str, str]]) -> ChatResult:
    return _chat_once(
        args.server,
        {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        },
        stream=args.stream,
    )


def _generate_reply(args, session: SessionState) -> bool:
    print("assistant> ", end="", flush=True)
    try:
        result = _request_assistant(args, session.messages)
    except urllib.error.URLError as e:
        print(f"\nRequest failed: {e}")
        return False
    except Exception as e:
        print(f"\nRequest failed: {e}")
        return False

    if not args.stream:
        print(result.text)
    session.messages.append({"role": "assistant", "content": result.text})
    _print_server_notes(result.meta)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000", type=str)
    parser.add_argument("--model", default="llaisys-qwen2", type=str)
    parser.add_argument("--max_tokens", default=64, type=int)
    parser.add_argument("--temperature", default=0.2, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--top_k", default=20, type=int)
    parser.add_argument("--repetition_penalty", default=1.0, type=float)
    parser.add_argument("--no_repeat_ngram_size", default=0, type=int)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--system",
        default=(
            "You are a concise assistant. Reply in the same language as the user. "
            "Give only the final answer. Do not reveal thinking steps. "
            "Do not repeat or paraphrase the user's question."
        ),
        type=str,
        help="System prompt for each new session",
    )
    args = parser.parse_args()

    sessions: Dict[str, SessionState] = {"default": _new_session(args.system)}
    active = "default"

    print("LLAISYS Chat CLI")
    print(HELP_TEXT)

    while True:
        try:
            user_text = input(f"[{active}]> ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            continue

        if user_text.startswith("/"):
            raw_parts = user_text.split(maxsplit=2)
            cmd = raw_parts[0].lower()
            arg1 = raw_parts[1] if len(raw_parts) > 1 else ""
            arg2 = raw_parts[2] if len(raw_parts) > 2 else ""
            session = sessions[active]

            if cmd == "/exit":
                break
            if cmd == "/help":
                print(HELP_TEXT)
                continue
            if cmd == "/list":
                for name in sorted(sessions.keys()):
                    marker = "*" if name == active else " "
                    print(f"{marker} {name}")
                continue
            if cmd == "/new":
                name = arg1.strip() or f"chat_{len(sessions)}"
                if name in sessions:
                    print("Session already exists.")
                    continue
                sessions[name] = _new_session(args.system)
                active = name
                print(f"Switched to session: {name}")
                continue
            if cmd == "/switch":
                name = arg1.strip()
                if not name or name not in sessions:
                    print("Unknown session.")
                    continue
                active = name
                print(f"Switched to session: {name}")
                continue
            if cmd == "/delete":
                name = arg1.strip()
                if not name or name not in sessions:
                    print("Unknown session.")
                    continue
                if name == "default" and len(sessions) == 1:
                    print("Cannot delete the last remaining session.")
                    continue
                del sessions[name]
                if active == name:
                    active = sorted(sessions.keys())[0]
                print(f"Deleted session: {name}")
                continue
            if cmd == "/clear":
                sessions[active] = _new_session(args.system)
                print(f"Cleared session: {active}")
                continue
            if cmd == "/history":
                limit = None
                if arg1:
                    try:
                        limit = max(1, int(arg1))
                    except ValueError:
                        print("Usage: /history [last_n_turns]")
                        continue
                _print_history(session.messages, limit=limit)
                continue
            if cmd == "/regen":
                target = _resolve_turn(session.messages, arg1 or "last")
                if target is None:
                    print("Nothing to regenerate.")
                    continue
                turn_no, user_idx, _ = target
                removed = len(session.messages) - (user_idx + 1)
                del session.messages[user_idx + 1 :]
                if removed > 0:
                    print(f"Regenerating from turn {turn_no}; removed {removed} later message(s).")
                if not _generate_reply(args, session):
                    continue
                continue
            if cmd == "/edit":
                if not arg1 or not arg2:
                    print("Usage: /edit <turn> <new text>")
                    continue
                target = _resolve_turn(session.messages, arg1)
                if target is None:
                    print("Unknown turn.")
                    continue
                turn_no, user_idx, _ = target
                session.messages[user_idx]["content"] = arg2
                removed = len(session.messages) - (user_idx + 1)
                del session.messages[user_idx + 1 :]
                print(f"Updated turn {turn_no}; removed {removed} later message(s).")
                if not _generate_reply(args, session):
                    continue
                continue

            print("Unknown command. Use /help.")
            continue

        session = sessions[active]
        session.messages.append({"role": "user", "content": user_text})
        if not _generate_reply(args, session):
            session.messages.pop()
            continue

    return 0


if __name__ == "__main__":
    sys.exit(main())
