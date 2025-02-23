import os
import librosa
import soundfile as sf
import pandas as pd
import numpy as np

# Configuration
INPUT_DIR = '../datasets' 
OUTPUT_DIR = '../preprocessed_data/processed_audio'
CSV_PATH = '../data_manifest.csv'
SAMPLE_RATE = 16000
CHUNK_DURATION = 10  # seconds

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define dataset structure: folder name maps to label
DATASETS = {
    'cry': 'crying',
    'control': 'speech',
    'scream': 'screaming'
}

manifest = []

def process_audio_file(file_path, label, output_dir):
    """Loads audio, splits into chunks, and saves."""
    # Load audio file
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    chunk_length = SAMPLE_RATE * CHUNK_DURATION
    num_chunks = int(np.ceil(len(audio) / chunk_length))

    
    for i in range(num_chunks):
        start = i * chunk_length
        end = start + chunk_length
        chunk = audio[start:end]
        if len(chunk) < chunk_length:
            # Pad with zeros if chunk is too short
            chunk = np.pad(chunk, (0, chunk_length - len(chunk)), mode='constant')
        
        # Save chunk to file
        file_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_chunk{i}.wav"
        output_path = os.path.join(output_dir, file_name)
        sf.write(output_path, chunk, SAMPLE_RATE)
        manifest.append({'file_path': output_path, 'label': label})

# Loop through each dataset folder and process files
for folder, label in DATASETS.items():
    folder_path = os.path.join(INPUT_DIR, folder)
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist!")
        continue
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            process_audio_file(os.path.join(folder_path, file), label, OUTPUT_DIR)


# Loop through each dataset folder and process files
for folder, label in DATASETS.items():
    folder_path = os.path.join(INPUT_DIR, folder)
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            process_audio_file(os.path.join(folder_path, file), label, OUTPUT_DIR)

# Save manifest to CSV
df = pd.DataFrame(manifest)
df.to_csv(CSV_PATH, index=False)
print(f"Data preprocessing completed. Manifest saved to {CSV_PATH}")
