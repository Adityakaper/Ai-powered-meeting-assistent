import os
import uuid
import json
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from summary_module import transcribe_and_summarize, chat_with_ai

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_FOLDER = os.path.join(BASE_DIR, "db")
ALLOWED_EXTENSIONS = {"mp3", "wav", "m4a", "mp4", "aac", "flac", "ogg", "wma", "webm"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

app = Flask(__name__, static_url_path="/static", static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "dev-secret-key"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_record(record_id, data):
    path = os.path.join(DB_FOLDER, f"{record_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_record(record_id):
    path = os.path.join(DB_FOLDER, f"{record_id}.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.route("/")
def home():
    last_id = session.get("last_record_id")
    record = load_record(last_id) if last_id else {}
    transcript = record.get("transcript", "")
    summary = record.get("summary", "")
    return render_template("index.html", transcript=transcript, summary=summary, current_page="home")


@app.route("/upload", methods=["POST"])
def upload_and_summarize():
    if "audio_file" not in request.files:
        return render_template("index.html", transcript="", summary="No file part in request.", current_page="home")

    file = request.files["audio_file"]
    if file.filename == "":
        return render_template("index.html", transcript="", summary="No file selected.", current_page="home")

    if not allowed_file(file.filename):
        return render_template("index.html", transcript="", summary="Unsupported file type.", current_page="home")

    orig_filename = secure_filename(file.filename)
    uid = uuid.uuid4().hex
    filename = f"{uid}_{orig_filename}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    record_id = uid
    record = {"id": record_id, "filename": filename, "filepath": filepath, "status": "processing"}
    save_record(record_id, record)
    session["last_record_id"] = record_id

    try:
        transcript, summary = transcribe_and_summarize(filepath)
        record["transcript"] = transcript
        record["summary"] = summary
        record["status"] = "done"
        save_record(record_id, record)
        return render_template("index.html", transcript=transcript, summary=summary, current_page="home")
    except Exception as e:
        record["status"] = "error"
        record["error"] = str(e)
        save_record(record_id, record)
        return render_template("index.html", transcript="", summary=f"Error: {e}", current_page="home")


# 🆕 CHAT PAGE ROUTE
@app.route("/chat", methods=["GET"])
def chat_page():
    return render_template("chat.html", current_page="chat")

@app.route("/about", methods=["GET"])
def about_page():
    return render_template("about.html", current_page="about")


@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json(silent=True) or {}
    user_message = str(data.get("message", "")).strip()
    if not user_message:
        return jsonify({"response": "Please enter a message."})

    record_id = session.get("last_record_id")
    record = load_record(record_id) if record_id else {}
    transcript = record.get("transcript", "")
    summary = record.get("summary", "")

    try:
        response = chat_with_ai(user_message, transcript, summary)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"response": f"Chat error: {e}"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
