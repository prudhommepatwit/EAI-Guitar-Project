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
    "A2" : "A2",
    "A3" : "A3",
    "A4" : "A4",
    "Asharp2" : "A#2",
    "Asharp3" : "A#3",
    "Asharp4" : "A#4",
    "B2" : "B2",
    "B3" : "B3",
    "B4" : "B4",
    "C3" : "C3",
    "C4" : "C4",
    "C5" : "C5",
    "Csharp3" : "C#3",
    "Csharp4" : "C#4",
    "Csharp5" : "C#5",
    "D2" : "D2",
    "D3" : "D3",
    "D4" : "D4",
    "D5" : "D5",
    "Dsharp2" : "D#2",
    "Dsharp3" : "D#3",
    "Dsharp4" : "D#4",
    "Dsharp5" : "D#5",
    "E3" : "E2",
    "E3" : "E3",
    "E3" : "E4",
    "E3" : "E5",
    "F3" : "F2",
    "F3" : "F3",
    "F3" : "F4",
    "F3" : "F5",
    "Fsharp2" : "F#2",
    "Fsharp3" : "F#3",
    "Fsharp4" : "F#4",
    "Fsharp5" : "F#5",
    "G2" : "G2",
    "G3" : "G3",
    "G4" : "G4",
    "G5" : "G5",
    "Csharp2" : "G#2",
    "Csharp3" : "G#3",
    "Csharp4" : "G#4",
    "Csharp5" : "G#5"
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

    #Temporarily returning just the id number
    #Casted from np.int32 to int
    return int(predicted_id)

    return id_to_note[int(predicted_id)]
