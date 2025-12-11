# import keras_hub
import tensorflow as tf
import numpy as np
import librosa
from pydub import AudioSegment
import noisereduce as nr
# communicate

MODEL_PATH = "noteClassifier.keras"
model = tf.keras.models.load_model(MODEL_PATH, custom_objects=None, compile=True)

#Update dictionary to take in (42? 32?) unique IDs and determine which note is which
id_to_note = {
    0 : "D",
    1 : "G",
    2 : "G#",
    3 : "N/A",
    4 : "N/A",
    5 : "E",
    6 : "N/A",
    7 : "G",
    8 : "F",
    9 : "N/A",
    10 : "E",
    11 : "N/A",
    12 : "N/A",
    13 : "A",
    14 : "N/A",
    15 : "N/A",
    16 : "N/A",
    17 : "A",
    18 : "N/A",
    19 : "N/A",
    20 : "N/A",
    21 : "A#",
    22 : "A#",
    23 : "N/A",
    24 : "N/A",
    25 : "G",
    26 : "C",
    27 : "N/A",
    28 : "N/A",
    29 : "B",
    30 : "B",
    31 : "N/A",
    32 : "E",
    33 : "C",
    34 : "C",
    35 : "N/A",
    36 : "N/A",
    37 : "D",
    38 : "N/A",
    39 : "E",
    40 : "F",
    41 : "D#",
    42 : "N/A"}

def convert_webm_to_wav(webm_path, wav_path):
    """Convert the webm input to wav format"""
    audio = AudioSegment.from_file(webm_path, format="webm")
    audio.export(wav_path, format="wav")
    return wav_path

def reduce_noise(y, sr):
    y_clean = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=0.8)
    return y_clean


def noteClassify(userAudioPath, target_sr=16000, max_len=16000):
    y, sr = librosa.load(userAudioPath, sr=target_sr)
    y = reduce_noise(y, sr)
    
    if len(y) > max_len:
        y = y[:max_len]
    else:
        y = np.pad(y, (0, max_len - len(y)), mode="constant")

    # Normalize
    y = y / np.max(np.abs(y))

    inputs = np.expand_dims(y, axis=0)
        
    

    outputs = model(inputs)
    predicted_id = np.argmax(outputs, axis=1)[0]

    # Temporarily returning just the id number
    # Casted from np.int32 to int
    # return int(predicted_id)
  

    return id_to_note[int(predicted_id)]
