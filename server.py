from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model on startup
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
logging.info("Model loaded successfully")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/embed", methods=["POST"])
@app.route("/embeddings", methods=["POST"])
@app.route("/v1/embeddings", methods=["POST"])
def embed():
    data = request.json
    inputs = data.get("input") or data.get("inputs") or data.get("text")
    
    if isinstance(inputs, str):
        inputs = [inputs]
    
    embeddings = model.encode(inputs, normalize_embeddings=True)
    
    # OpenAI-compatible format
    embedding_list = []
    for idx, emb in enumerate(embeddings.tolist()):
        embedding_list.append({
            "object": "embedding",
            "embedding": emb,
            "index": idx
        })
    
    return jsonify({
        "object": "list",
        "data": embedding_list,
        "model": "BAAI/bge-small-en-v1.5",
        "usage": {
            "prompt_tokens": sum(len(str(inp).split()) for inp in inputs),
            "total_tokens": sum(len(str(inp).split()) for inp in inputs)
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
