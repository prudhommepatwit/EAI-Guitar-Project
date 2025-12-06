# import keras_hub
import tensorflow as tf
import torch
# communicate

MODEL_PATH = "noteClassifier.keras"
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

def noteClassify(userAudio):
    inputs = userAudio

    with torch.no_grad():
        outputs = model(inputs)

    predicted_id = torch.argmax(outputs.logits, dim=1).item()
    predicted_note = id_to_note[predicted_id]

    return predicted_note