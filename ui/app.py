"""
Layer 6: Flask UI
-----------------
A minimal chatbot web app. One endpoint: POST /chat
The front end (index.html) sends a JSON question, gets a JSON answer.

Wired together:
  request -> semantic cache check -> agent -> cache store -> response
"""

import os
import sys
import time
import uuid
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import FLASK_PORT, FLASK_DEBUG

from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS

from agent.rag_agent import build_agent, chat, stream_chat
from cache.semantic_cache import SemanticCache

app = Flask(__name__, static_folder="static")
CORS(app)

# Build agent and cache once at startup — expensive to rebuild per request
print("Loading agent (this may take ~30s on first run for embeddings)...")
_agent = build_agent()
_cache = SemanticCache()
print("Agent ready.")

# Each browser session gets its own thread_id so memory is isolated
_sessions: dict[str, str] = {}


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not question:
        return jsonify({"error": "question is required"}), 400

    start = time.time()

    # Semantic cache check
    cached = _cache.get(question)
    if cached:
        duration = round(time.time() - start, 3)
        return jsonify({"answer": cached, "session_id": session_id, "cached": True, "duration_s": duration})

    # Agent call
    answer = chat(_agent, question, session_id)
    duration = round(time.time() - start, 3)

    # Store in cache
    _cache.set(question, answer)
    _cache.evict_expired()

    return jsonify({"answer": answer, "session_id": session_id, "cached": False, "duration_s": duration})


@app.route("/chat/stream", methods=["POST"])
def chat_stream_endpoint():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not question:
        return jsonify({"error": "question is required"}), 400

    start = time.time()

    # Cache hit — send a single SSE event and close immediately
    cached = _cache.get(question)
    if cached:
        duration = round(time.time() - start, 3)
        def cached_stream():
            payload = json.dumps({"answer": cached, "cached": True,
                                  "session_id": session_id, "duration_s": duration})
            yield f"data: {payload}\n\n"
        return Response(stream_with_context(cached_stream()), mimetype="text/event-stream")

    # Cache miss — stream tokens from the agent
    def generate():
        for event_type, content in stream_chat(_agent, question, session_id):
            if event_type == "token":
                yield f"data: {json.dumps({'token': content})}\n\n"
            elif event_type == "status":
                yield f"data: {json.dumps({'status': content})}\n\n"
            elif event_type == "done":
                duration = round(time.time() - start, 3)
                _cache.set(question, content)
                _cache.evict_expired()
                payload = json.dumps({"done": True, "session_id": session_id,
                                      "duration_s": duration})
                yield f"data: {payload}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "cache_size": _cache.size})


if __name__ == "__main__":
    app.run(port=FLASK_PORT, debug=FLASK_DEBUG)
