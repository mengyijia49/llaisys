import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from typing import Dict, List

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def _sanitize_console_text(text: str) -> str:
    text = ANSI_ESCAPE_RE.sub("", text)
    text = CONTROL_CHAR_RE.sub("", text)
    return text


def _post_json(url: str, payload: Dict):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    return urllib.request.urlopen(req)


def _chat_once(server: str, payload: Dict, stream: bool) -> str:
    endpoint = f"{server.rstrip('/')}/v1/chat/completions"
    payload = dict(payload)
    payload["stream"] = stream

    if not stream:
        with _post_json(endpoint, payload) as resp:
            body = resp.read().decode("utf-8")
        obj = json.loads(body)
        return _sanitize_console_text(obj["choices"][0]["message"]["content"])

    chunks: List[str] = []
    with _post_json(endpoint, payload) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            evt = json.loads(data)
            delta = evt["choices"][0].get("delta", {})
            piece = _sanitize_console_text(delta.get("content", ""))
            if piece:
                print(piece, end="", flush=True)
                chunks.append(piece)
    print()
    return "".join(chunks)


def _new_session(system_prompt: str) -> List[Dict[str, str]]:
    if system_prompt:
        return [{"role": "system", "content": system_prompt}]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8000", type=str)
    parser.add_argument("--model", default="llaisys-qwen2", type=str)
    parser.add_argument("--max_tokens", default=128, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top_p", default=0.9, type=float)
    parser.add_argument("--top_k", default=40, type=int)
    parser.add_argument("--repetition_penalty", default=1.1, type=float)
    parser.add_argument("--no_repeat_ngram_size", default=3, type=int)
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

    sessions: Dict[str, List[Dict[str, str]]] = {"default": _new_session(args.system)}
    active = "default"

    print("LLAISYS Chat CLI")
    print("Commands: /help /new [name] /switch <name> /list /clear /regen /exit")

    while True:
        try:
            user_text = input(f"[{active}]> ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            continue

        if user_text.startswith("/"):
            parts = user_text.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/exit":
                break
            if cmd == "/help":
                print("Commands: /help /new [name] /switch <name> /list /clear /regen /exit")
                continue
            if cmd == "/list":
                print("Sessions:", ", ".join(sorted(sessions.keys())))
                continue
            if cmd == "/new":
                name = arg.strip() or f"chat_{len(sessions)}"
                sessions[name] = _new_session(args.system)
                active = name
                print(f"Switched to session: {name}")
                continue
            if cmd == "/switch":
                name = arg.strip()
                if not name or name not in sessions:
                    print("Unknown session.")
                    continue
                active = name
                print(f"Switched to session: {name}")
                continue
            if cmd == "/clear":
                sessions[active] = _new_session(args.system)
                print(f"Cleared session: {active}")
                continue
            if cmd == "/regen":
                msgs = sessions[active]
                if msgs and msgs[-1]["role"] == "assistant":
                    msgs.pop()
                if not msgs or msgs[-1]["role"] != "user":
                    print("Nothing to regenerate.")
                    continue
                print("assistant> ", end="", flush=True)
                try:
                    text = _chat_once(
                        args.server,
                        {
                            "model": args.model,
                            "messages": msgs,
                            "max_tokens": args.max_tokens,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "top_k": args.top_k,
                            "repetition_penalty": args.repetition_penalty,
                            "no_repeat_ngram_size": args.no_repeat_ngram_size,
                        },
                        stream=args.stream,
                    )
                except urllib.error.URLError as e:
                    print(f"\nRequest failed: {e}")
                    continue
                if not args.stream:
                    print(text)
                msgs.append({"role": "assistant", "content": text})
                continue

            print("Unknown command. Use /help.")
            continue

        msgs = sessions[active]
        msgs.append({"role": "user", "content": user_text})
        print("assistant> ", end="", flush=True)
        try:
            text = _chat_once(
                args.server,
                {
                    "model": args.model,
                    "messages": msgs,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                    "no_repeat_ngram_size": args.no_repeat_ngram_size,
                },
                stream=args.stream,
            )
        except urllib.error.URLError as e:
            msgs.pop()
            print(f"\nRequest failed: {e}")
            continue
        except Exception as e:
            msgs.pop()
            print(f"\nRequest failed: {e}")
            continue

        if not args.stream:
            print(text)
        msgs.append({"role": "assistant", "content": text})

    return 0


if __name__ == "__main__":
    sys.exit(main())
