import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Configuration
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 3
SAMPLE_RATE = 16000
MANIFEST_PATH = '../data_manifest.csv'
MODEL_SAVE_PATH = '../models/yamnet_finetuned.h5'

# Load dataset
manifest = pd.read_csv(MANIFEST_PATH)
label_to_int = {'crying': 0, 'screaming': 1, 'speech': 2}
manifest = manifest[manifest['label'].isin(label_to_int.keys())].copy()
manifest['label_int'] = manifest['label'].map(label_to_int)

# Stratified split
total_samples = len(manifest)
test_count = int(0.15 * total_samples)
manifest = manifest.sort_index()
test_manifest = manifest.tail(test_count)
test_file_paths = test_manifest["file_path"].values
test_labels = test_manifest["label_int"].values

# Audio Preprocessing
def load_audio(file_path):
    audio, _ = librosa.load(file_path.decode('utf-8'), sr=SAMPLE_RATE)
    return np.array(audio, dtype=np.float32)

def preprocess_audio(file_path, label):
    audio = tf.numpy_function(func=load_audio, inp=[file_path], Tout=tf.float32)
    audio.set_shape([None])
    return audio, label

# Create tf.data Datasets
file_paths_full = manifest['file_path'].values
labels_full = manifest['label_int'].values
dataset = tf.data.Dataset.from_tensor_slices((file_paths_full, labels_full))
dataset = dataset.map(preprocess_audio, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000)

train_ds = dataset.take(int(0.7 * total_samples)).padded_batch(BATCH_SIZE, padded_shapes=([None], []))
val_ds = dataset.skip(int(0.7 * total_samples)).take(int(0.15 * total_samples)).padded_batch(BATCH_SIZE, padded_shapes=([None], []))
test_ds = tf.data.Dataset.from_tensor_slices((test_file_paths, test_labels)).map(preprocess_audio).padded_batch(BATCH_SIZE, padded_shapes=([None], []))

# Load YAMNet Model
yamnet_url = "https://tfhub.dev/google/yamnet/1"
yamnet_layer = hub.KerasLayer(yamnet_url, trainable=False, signature="serving_default", output_key="output_1")

def get_average_embedding(waveform):
    return tf.reduce_mean(yamnet_layer({"waveform": waveform}), axis=0)

@tf.keras.utils.register_keras_serializable(package="Custom", name="average_embedding_layer")
def average_embedding_layer(x):
    return tf.map_fn(get_average_embedding, x, fn_output_signature=tf.float32)

# Build Model
def build_model():
    inputs = tf.keras.Input(shape=(None,), dtype=tf.float32, name="audio_input")
    embeddings = tf.keras.layers.Lambda(average_embedding_layer)(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(embeddings)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs)

model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)

# Save Training Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss", marker="o")
plt.plot(history.history.get("val_loss", []), label="Val Loss", marker="o")
plt.xlabel("Epoch"), plt.ylabel("Loss"), plt.title("Loss Curve"), plt.legend(), plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Accuracy", marker="o")
plt.plot(history.history.get("val_accuracy", []), label="Val Accuracy", marker="o")
plt.xlabel("Epoch"), plt.ylabel("Accuracy"), plt.title("Accuracy Curve"), plt.legend(), plt.grid(True)

plt.tight_layout(), plt.savefig("loss_accuracy_curves_yamnet.pdf"), plt.close()

# Evaluate on Test Set
all_test_preds, all_test_labels, all_test_probs = [], [], []
for batch in test_ds:
    inputs, labels = batch
    outputs = model(inputs, training=False)
    preds = tf.argmax(outputs, axis=-1)
    all_test_preds.extend(preds.numpy())
    all_test_labels.extend(labels.numpy())
    all_test_probs.extend(outputs.numpy())

test_acc = np.mean(np.array(all_test_preds) == np.array(all_test_labels))
print(f"Test Accuracy: {test_acc:.4f}")

# Classification Report
report_text = classification_report(all_test_labels, all_test_preds, target_names=list(label_to_int.keys()))
cm = confusion_matrix(all_test_labels, all_test_preds)

# Save Classification Report
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')
ax.text(0.01, 0.5, f"Classification Report:\n\n{report_text}\n\nConfusion Matrix:\n{cm}", fontsize=10, verticalalignment='center', transform=ax.transAxes)
plt.title("Classification Report & Confusion Matrix")
plt.savefig("classification_report_yamnet.pdf"), plt.close()

# ROC Curves
all_test_labels_bin = label_binarize(all_test_labels, classes=[0, 1, 2])
fpr, tpr, roc_auc = {}, {}, {}
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(all_test_labels_bin[:, i], np.array(all_test_probs)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i, color in zip(range(NUM_CLASSES), ['blue', 'red', 'green']):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"{list(label_to_int.keys())[i]} (AUC = {roc_auc[i]:0.2f})")
plt.plot([0, 1], [0, 1], 'k--'), plt.xlabel("False Positive Rate"), plt.ylabel("True Positive Rate"), plt.legend()
plt.savefig("roc_curves_yamnet.pdf"), plt.close()

# Save Model
model.save(MODEL_SAVE_PATH)
print("Training complete. Model and reports saved.")