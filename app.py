from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, Response, send_file
from flask_socketio import SocketIO, emit
import webbrowser
from threading import Timer
from predict import noteClassify, convert_webm_to_wav
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/listen", methods=["POST"])
def listen():
    
    file = request.files["audio"]
    print("Received file:", file.filename)

    os.makedirs("uploads", exist_ok=True)
    webm_path = os.path.join("uploads", file.filename)
    file.save(webm_path)
    
    # Debug
    print("Saved to:", webm_path)
    
    wav_path = webm_path.replace(".webm", ".wav")
    convert_webm_to_wav(webm_path, wav_path)
    
    #classification method from the model running script(user_input)
    predicted_note = noteClassify(wav_path)
    return jsonify({"note": predicted_note}) #user_input reference section)

if __name__ == '__main__':
    Timer(1,lambda: webbrowser.open_new("http://127.0.0.1:8080/")).start()
    # socketio.run(app)
    socketio.run(app, debug=True)
