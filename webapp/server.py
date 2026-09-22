"""A small local web chat interface for a trained classroom nanoGPT model.

Reuses the exact same inference code as chat.py / run_evals.py — no training,
no vocabulary changes, no writes to any corpus. Each prompt starts a fresh
48-token context, same as the terminal interface.

Run:
    pip install -r webapp/requirements.txt
    python webapp/server.py --model llm_runs/<run>/model.pt --port 5050
"""
import argparse
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from run_evals import generate_reply, load_model, model_hash  # noqa: E402

DEFAULT_MODEL = ROOT / "llm_runs" / "20260922T173848_430218Z" / "model.pt"
LOG_PATH = Path(__file__).resolve().parent / "webapp_chat_log.json"

app = Flask(__name__)
state = {"model": None, "vocabulary": None, "saved": None, "turns": 0}


def load_log():
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text())
    return []


def append_log(entry):
    log = load_log()
    log.append(entry)
    LOG_PATH.write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/info")
def info():
    model = state["model"]
    return jsonify({
        "model_sha256": model_hash(model),
        "completed_steps": state["saved"].get("completed_steps"),
        "vocabulary_size": len(state["vocabulary"]),
        "block_size": model.config.block_size,
        "model_path": state["model_path"],
    })


@app.post("/api/chat")
def chat():
    payload = request.get_json(force=True, silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Empty prompt."}), 400
    seed = 2026 + state["turns"]
    reply = generate_reply(state["model"], state["vocabulary"], prompt, seed=seed)
    state["turns"] += 1
    append_log({"prompt": prompt, "seed": seed, **reply})
    return jsonify(reply)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    model, vocabulary, saved = load_model(args.model)
    state.update(model=model, vocabulary=vocabulary, saved=saved, model_path=str(args.model))
    print(f"Loaded {args.model} | steps={saved.get('completed_steps')} | vocab={len(vocabulary)} "
          f"| hash={model_hash(model)[:12]}...")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
