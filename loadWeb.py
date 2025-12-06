from flask import Flask, render_template, request, jsonify, url_for, redirect, flash, Response, send_file
from flask_socketio import SocketIO, emit
import webbrowser
from threading import Timer
from predict import noteClassify

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route("/listen", methods=["POST"])
def listen():
    userAudio = request.files["audio"].read()
    with open("userAudio.webm", "wb") as f:
        f.write(userAudio)

    response = noteClassify(userAudio)#classification method from the model running script(user_input)

    return render_template("index.html", response=response) #user_input reference section)

if __name__ == '__main__':
    Timer(1,lambda: webbrowser.open_new("http://127.0.0.1:5000/")).start()
    socketio.run(app)