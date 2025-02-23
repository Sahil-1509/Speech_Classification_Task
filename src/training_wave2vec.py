import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, AdamW
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Configuration
BATCH_SIZE = 8
EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 1e-5
NUM_CLASSES = 3
SAMPLE_RATE = 16000
MANIFEST_PATH = '../data_manifest.csv'
MODEL_SAVE_PATH = '../models/wav2vec2_finetuned1'

# Load dataset
manifest = pd.read_csv(MANIFEST_PATH)
label_to_int = {'crying': 0, 'screaming': 1, 'speech': 2}
manifest = manifest[manifest['label'].isin(label_to_int.keys())].copy()
manifest['label_int'] = manifest['label'].map(label_to_int)

def add_random_noise(audio, noise_factor=0.005):
    return (audio + noise_factor * np.random.randn(len(audio))).astype(np.float32)

class AudioDataset(Dataset):
    def __init__(self, df, processor, sample_rate=SAMPLE_RATE, augment=False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.sample_rate = sample_rate
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            audio, _ = librosa.load(row['file_path'], sr=self.sample_rate)
            if self.augment:
                audio = add_random_noise(audio)
        except Exception:
            audio = np.zeros(1, dtype=np.float32)
        inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        return {k: v.squeeze(0) for k, v in inputs.items()}, torch.tensor(row['label_int'], dtype=torch.long)

# Model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=NUM_CLASSES)

# Data splitting
dataset_all = AudioDataset(manifest, processor, augment=True)
train_size = int(0.7 * len(dataset_all))
val_size = int(0.15 * len(dataset_all))
test_size = len(dataset_all) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset_all, [train_size, val_size, test_size])

val_dataset.dataset.augment = False
test_dataset.dataset.augment = False

# Dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    input_values = [item[0]['input_values'] for item in batch]
    attention_masks = [item[0].get('attention_mask', torch.ones_like(item[0]['input_values'])) for item in batch]
    labels = torch.stack([item[1] for item in batch])
    return {"input_values": torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True).to(device),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True).to(device),
            "labels": labels.to(device)}

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# Training setup
model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
best_val_loss, patience_counter = float('inf'), 0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_train, correct_train, running_loss = 0, 0, 0.0
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=-1)
        correct_train += (preds == batch["labels"]).sum().item()
        total_train += batch["labels"].size(0)

    avg_train_loss, train_acc = running_loss / len(train_loader), correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    total_val, correct_val, running_val_loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(**batch)
            running_val_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)
            correct_val += (preds == batch["labels"]).sum().item()
            total_val += batch["labels"].size(0)
    avg_val_loss, val_acc = running_val_loss / len(val_loader), correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}.")
            break

# Save training plots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Val Loss', marker='o')
plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.title('Loss Curve'), plt.legend(), plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', marker='o')
plt.plot(val_accuracies, label='Val Accuracy', marker='o')
plt.xlabel('Epoch'), plt.ylabel('Accuracy'), plt.title('Accuracy Curve'), plt.legend(), plt.grid(True)
plt.tight_layout(), plt.savefig("loss_accuracy_curves.pdf"), plt.close()

# Test Evaluation
model.eval()
all_test_preds, all_test_labels, all_test_probs = [], [], []
with torch.no_grad():
    for batch in test_loader:
        outputs = model(**batch)
        probs = torch.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        all_test_preds.extend(preds.cpu().numpy())
        all_test_labels.extend(batch["labels"].cpu().numpy())
        all_test_probs.extend(probs.cpu().numpy())

test_acc = np.mean(np.array(all_test_preds) == np.array(all_test_labels))
print(f"Test Accuracy: {test_acc:.4f}")

# Classification report & confusion matrix
report_text = classification_report(all_test_labels, all_test_preds, target_names=list(label_to_int.keys()))
cm = confusion_matrix(all_test_labels, all_test_preds)

fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')
ax.text(0.01, 0.5, f"Classification Report:\n\n{report_text}\n\nConfusion Matrix:\n{cm}", fontsize=10, verticalalignment='center', transform=ax.transAxes)
plt.title("Classification Report & Confusion Matrix")
plt.savefig("classification_report.pdf"), plt.close()

# Save ROC Curves
all_test_labels_bin = label_binarize(all_test_labels, classes=[0, 1, 2])
fpr, tpr, roc_auc = {}, {}, {}
for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(all_test_labels_bin[:, i], np.array(all_test_probs)[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(8, 6))
for i, color in zip(range(NUM_CLASSES), ['blue', 'red', 'green']):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f"{list(label_to_int.keys())[i]} (AUC = {roc_auc[i]:0.2f})")
plt.plot([0, 1], [0, 1], 'k--'), plt.xlabel("False Positive Rate"), plt.ylabel("True Positive Rate"), plt.legend()
plt.savefig("roc_curves.pdf"), plt.close()

# Save model
model.save_pretrained(MODEL_SAVE_PATH)
processor.save_pretrained(MODEL_SAVE_PATH)
print("Training complete. Model and reports saved.")