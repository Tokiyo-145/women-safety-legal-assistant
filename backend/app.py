from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from logic.search import find_law
import os

# Path to frontend folder
FRONTEND_FOLDER = os.path.join(os.path.dirname(__file__), "..", "frontend1")

app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path="")
CORS(app)

# Serve UI on localhost
@app.route("/")
def serve_ui():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()

    if not data or "problem" not in data:
        return jsonify({"error": "Problem description required"}), 400

    problem = data["problem"]

    # 👇 Now returns list of laws
    results = find_law(problem, top_k=3)

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)