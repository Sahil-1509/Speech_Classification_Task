import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub

# Configuration
SAMPLE_RATE = 16000
MANIFEST_PATH = "../data_manifest.csv"
YAMNET_MODEL_PATH = "../models/yamnet_finetuned.h5"
WAV2VEC2_MODEL_PATH = "../models/wav2vec2_finetuned1"
LABEL_TO_INT = {"crying": 0, "screaming": 1, "speech": 2}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YAMNet model with custom layer
yamnet_url = "https://tfhub.dev/google/yamnet/1"
yamnet_layer = hub.KerasLayer(yamnet_url, trainable=False, signature="serving_default", output_key="output_1")

def get_average_embedding(waveform):
    embeddings = yamnet_layer({"waveform": waveform})
    return tf.reduce_mean(embeddings, axis=0)

@tf.keras.utils.register_keras_serializable(package="Custom", name="average_embedding_layer")
def average_embedding_layer(x):
    return tf.map_fn(get_average_embedding, x, fn_output_signature=tf.float32)

# Load data and perform stratified sampling
manifest = pd.read_csv(MANIFEST_PATH)
manifest = manifest[manifest["label"].isin(LABEL_TO_INT.keys())].copy()
manifest["label_int"] = manifest["label"].map(LABEL_TO_INT)

train_manifest, test_manifest = train_test_split(manifest, test_size=0.15, stratify=manifest["label_int"], random_state=42)

test_file_paths = test_manifest["file_path"].values
test_labels = test_manifest["label_int"].values

# Audio loading functions
def load_audio(file_path):
    file_path = file_path.decode("utf-8") if isinstance(file_path, bytes) else file_path
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    return np.array(audio, dtype=np.float32)

def tf_load_audio(file_path, label):
    audio = tf.numpy_function(func=load_audio, inp=[file_path], Tout=tf.float32)
    audio.set_shape([None])
    return audio, label

# Create a test dataset
test_dataset = tf.data.Dataset.from_tensor_slices((test_file_paths, test_labels))
test_dataset = test_dataset.map(tf_load_audio, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.padded_batch(32, padded_shapes=([None], []))
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

# Load models
yamnet_model = tf.keras.models.load_model(YAMNET_MODEL_PATH, custom_objects={"average_embedding_layer": average_embedding_layer})
wav2vec2_processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_PATH)
wav2vec2_model = Wav2Vec2ForSequenceClassification.from_pretrained(WAV2VEC2_MODEL_PATH, num_labels=len(LABEL_TO_INT))
wav2vec2_model.to(DEVICE)
wav2vec2_model.eval()

# Ensemble Inference
ensemble_preds, ensemble_probs, true_labels = [], [], []

for _, row in test_manifest.iterrows():
    file_path, true_label = row["file_path"], row["label_int"]
    
    try:
        audio = load_audio(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        continue

    # YAMNet inference
    audio_tf = np.expand_dims(audio, axis=0)
    yamnet_probs = yamnet_model.predict(audio_tf).squeeze(0)

    # Wav2Vec2 inference
    inputs = wav2vec2_processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = wav2vec2_model(**inputs)
    wav2vec2_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy().squeeze(0)

    # Ensemble by averaging probabilities
    combined_probs = (yamnet_probs + wav2vec2_probs) / 2.0
    pred_class = int(np.argmax(combined_probs))

    ensemble_preds.append(pred_class)
    ensemble_probs.append(combined_probs)
    true_labels.append(true_label)

true_labels = np.array(true_labels)
ensemble_preds = np.array(ensemble_preds)
ensemble_probs = np.array(ensemble_probs)

# Classification Report & Confusion Matrix
report_text = classification_report(true_labels, ensemble_preds, target_names=list(LABEL_TO_INT.keys()), zero_division=0)
cm = confusion_matrix(true_labels, ensemble_preds)

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis("off")
ax.text(0.01, 0.5, f"Ensemble Classification Report:\n\n{report_text}\n\nConfusion Matrix:\n{cm}", fontsize=10, verticalalignment="center", transform=ax.transAxes)
plt.title("Ensemble Classification Report & Confusion Matrix")
plt.savefig("ensemble_classification_report.pdf")
plt.close()

# ROC Curves
y_true_bin = label_binarize(true_labels, classes=[0, 1, 2])
n_classes = y_true_bin.shape[1]
fpr, tpr, roc_auc = {}, {}, {}

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], ensemble_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
colors = ["blue", "red", "green"]
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"Class {i} ({list(LABEL_TO_INT.keys())[i]}) (AUC = {roc_auc[i]:0.2f})")

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Ensemble Model")
plt.legend(loc="lower right")
plt.savefig("ensemble_roc_curves.pdf")
plt.close()

print("Ensemble evaluation complete.")
print("Classification report saved to ensemble_classification_report.pdf")
print("ROC curves saved to ensemble_roc_curves.pdf")