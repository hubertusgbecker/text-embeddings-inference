"""
OpenAI-compatible embeddings wrapper for text-embeddings-inference.

Proxies /v1/embeddings requests to the text-embeddings-inference backend,
translating between API formats as needed.

Expected by OpenClaw memorySearch on port 11435.
"""

import logging
import os

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://text-embeddings-inference:80")


@app.route("/health", methods=["GET"])
def health():
    """Health check — also verifies backend is reachable."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.ok:
            return jsonify({"status": "ok", "backend": "ok"})
        return jsonify({"status": "degraded", "backend": r.status_code}), 503
    except Exception as e:
        return jsonify({"status": "degraded", "backend": str(e)}), 503


@app.route("/v1/embeddings", methods=["POST"])
@app.route("/embeddings", methods=["POST"])
def embeddings():
    """Forward embedding requests to the TEI backend in OpenAI-compatible format."""
    data = request.json or {}

    # Forward to TEI backend
    try:
        resp = requests.post(
            f"{BACKEND_URL}/v1/embeddings",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        return (resp.content, resp.status_code, {"Content-Type": "application/json"})
    except requests.exceptions.ConnectionError:
        return jsonify({"error": f"Cannot connect to backend at {BACKEND_URL}"}), 502
    except requests.exceptions.Timeout:
        return jsonify({"error": "Backend request timed out"}), 504


@app.route("/v1/models", methods=["GET"])
@app.route("/models", methods=["GET"])
def models():
    """Return available models — needed by some OpenAI-compatible clients."""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "bge-small-en-v1.5",
                "object": "model",
                "created": 1700000000,
                "owned_by": "BAAI",
            }
        ],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
