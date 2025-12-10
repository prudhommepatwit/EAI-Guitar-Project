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
    0 : "A2",
    1 : "A3",
    2 : "A4",
    3 : "A#2",
    4 : "A#3",
    5 : "A#4",
    6 : "B2",
    7 : "B3",
    8 : "B4",
    9 : "C3",
    10 : "C4",
    11 : "C5",
    12 : "C#3",
    13 : "C#4",
    14 : "C#5",
    15 : "D2",
    16 : "D3",
    17 : "D4",
    18 : "D5",
    19 : "D#2",
    20 : "D#3",
    21 : "D#4",
    22 : "D#5",
    23 : "E2",
    24 : "E3",
    25 : "E4",
    26 : "E5",
    27 : "F2",
    28 : "F3",
    29 : "F4",
    30 : "F5",
    31 : "F#2",
    32 : "F#3",
    33 : "F#4",
    34 : "F#5",
    35 : "G2",
    36 : "G3",
    37 : "G4",
    38 : "G5",
    39 : "G#2",
    40 : "G#3",
    41 : "G#4",
    42 : "G#5"
}

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
