# import keras_hub
import tensorflow as tf
import numpy as np
import librosa
from pydub import AudioSegment
# communicate

MODEL_PATH = "noteClassifier.keras"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=None, compile=True)

#Update dictionary to take in (42? 32?) unique IDs and determine which note is which
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
    11 : "G#/Ab",
    12 : "",
    13: "",
    14 : "",
    15 : "",
    16 : "",
    17 : "",
    18 : "",
    19 : "",
    20 : "",
    21 : "",
    22 : "",
    23 : "",
    24 : "",
    25 : "",
    26 : "",
    27 : "",
    28: "",
    29: "",
    30 : "",
    31 : "",
    32 : ""
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
    y = y / np.max(np.abs(y))

    inputs = np.expand_dims(y, axis=0)
        
    

    outputs = model(inputs)
    predicted_id = np.argmax(outputs, axis=1)[0]

    #Temporarily returning just the id number
    #Casted from np.int32 to int
    return int(predicted_id)

    return id_to_note[int(predicted_id)]