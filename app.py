from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, Response, send_file
from flask_socketio import SocketIO, emit
from predict import noteClassify, convert_webm_to_wav
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode="threading")


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/listen", methods=["POST"])
def listen():
    file = request.files["audio"]
    os.makedirs("uploads", exist_ok=True)
    
    webm_path = os.path.join("uploads", file.filename)
    file.save(webm_path)
    
    wav_path = webm_path.replace(".webm", ".wav")
    convert_webm_to_wav(webm_path, wav_path)
    
    #classification method from the model running script(user_input)
    predicted_note = noteClassify(wav_path)
    return jsonify({"note": predicted_note})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    socketio.run(app, debug=True)
