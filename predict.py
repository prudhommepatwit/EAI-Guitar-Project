# import keras_hub
import tensorflow as tf
import numpy as np
import librosa
from pydub import AudioSegment
# communicate

MODEL_PATH = "C:/Users/School/OneDrive - Wentworth Institute of Technology/Documents/2025/Embedded AI/EAI-Guitar-Project/noteClassifier.keras"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=None, compile=True)

id_to_note = {
    0 : "A",
    1 : "A#/Bb",
    2 : "B",
    3 : "C",
    4 : "C#/Db",
    5 : "D",
    6 : "D#/Eb",
    7 : "E",
    8 : "F",
    9 : "F#/Gb",
    10 : "G",
    11 : "G#/Ab"

}

def convert_webm_to_wav(webm_path, wav_path):
    """Convert the webm input to wav format"""
    audio = AudioSegment.from_file(webm_path, format="webm")
    audio.export(wav_path, format="wav")
    return wav_path

def noteClassify(userAudioPath, target_sr=16000, max_len=16000):
    y, sr = librosa.load(userAudioPath, sr=target_sr)
    
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)), mode="constant")

    # Normalize
    y = y / np.max(no.abs(y))

    inputs = np.expand_dims(y, axis=0)
        
    

    outputs = model(inputs)
    predicted_id = np.argmax(outputs, axis=1)[0]

    return id_to_note[predicted_id]