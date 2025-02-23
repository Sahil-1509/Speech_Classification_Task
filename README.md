# Ensemble Model for Detecting Infant Cries, Screams, and Normal Utterances

## Project Overview
This project focuses on **detecting infant cries, screams, and normal utterances** from audio data. We employ two distinct deep learning architectures—**YAMNet** (based on a MobileNet feature extractor for audio) and **Wav2Vec2** (a transformer-based model from the Hugging Face ecosystem)—and combine their predictions via an **ensemble** approach. The end goal is to **deploy** this system in a [Temporal](https://temporal.io/) workflow that can handle real-time audio streams.

---

## Assignment Objectives & Tasks
According to the assignment:

1. **Select an appropriate number of samples** in the experimental (cry, scream) and control (speech, music, other sounds) datasets.
2. **Preprocess** audio data to ensure consistency (e.g., sampling rate, bit depth).
3. **Train** two models:
   - Fine-tuned **YAMNet** 
   - Fine-tuned **Wav2Vec2**
4. **Develop an Ensemble** of the two models.
5. **Evaluate** models using metrics like accuracy, precision, recall, F1-score, confusion matrices, ROC curves, etc.
6. **Deploy** the final ensemble in a **Temporal workflow** to manage tasks such as:
   - Receiving audio input
   - Running the classification
   - Storing/managing results
7. **Deliverables** include code, readme, training graphs, and inference instructions.

---

## Repository Structure

my_project/
│
├── data_manifest.csv
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── src/
│   ├── init.py
│   ├── preprocess.py
│   ├── training.py
│   ├── training_wave2vec2.py
│   ├── testing_YAMNet.py
│   ├── ensemblemodel.py     # Contains the final ensemble inference logic
│   ├── pipeline.ipynb       # Demonstration or pipeline notebook
│   └── …
│
├── models/
│   ├── yamnet_finetuned.h5
│   ├── wav2vec2_finetuned/  # Directory containing PyTorch model checkpoint
│   └── …
│
├── notebooks/
│   ├── code_1.ipynb
│   └── …
│
├── preprocessed_data/
│   └── …
│
├── processed_audio/
│   └── …
│
├── temporal_workflow/
│   ├── worker.py            # Defines Temporal worker & activities
│   ├── workflow.py          # Defines Temporal workflow
│   └── client.py            # Triggers workflow executions
│
└── plots/
├── classification_report_yamnet.pdf
├── ensemble_classification_report.pdf
├── ensemble_roc_curves.pdf
├── loss_accuracy_curves_yamnet.pdf
├── loss_accuracy_curves_wav2vec2.pdf
└── …

---

## Data Acquisition & Preprocessing

### Dataset Selection
We use several open-source datasets for:
- **Infant Cry**: [Infant Cry and Snoring Detection Dataset (ICSD)](https://arxiv.org/), [Infant Cry Audio Corpus (Kaggle)](https://www.kaggle.com/), etc.
- **Screams**: [Human Screaming Detection Dataset (Kaggle)](https://www.kaggle.com/), [Screaming Audio Clips from AudioSet](https://research.google.com/audioset/)
- **Normal Utterances**: [Common Voice Dataset](https://commonvoice.mozilla.org/), [LibriSpeech](http://www.openslr.org/12/), children’s speech from AudioSet, etc.

### Data Preparation
1. **Consistent Sampling Rate**: We resample all audio to 16kHz.
2. **Segmentation**: Each audio file is segmented into short clips (5-15 seconds).
3. **Labeling**: Segments are labeled as `crying`, `screaming`, or `normal`.
4. **Metadata**: A CSV (`data_manifest.csv`) keeps track of `file_path` and `label`.

### Sample Size Justification
- We aimed for a balanced dataset with enough examples of each class (cry, scream, normal speech). 
- **Experimental**: cry and scream.  
- **Control**: normal speech, music, ambient indoor sounds.
- To ensure each class is well-represented, we used **stratified sampling** to select training/validation/test splits. This helps maintain class distribution in each split.

---

## Model Training

### YAMNet Fine-Tuning
- **Base Model**: [YAMNet](https://tfhub.dev/google/yamnet/1) which is pretrained on AudioSet.
- **Fine-Tuning**: We replace the classification layer to output 3 classes and freeze/unfreeze select layers.
- **Implementation**: In `src/training.py` (or a separate script), we load YAMNet from TF Hub, adapt final layers, and train on our labeled segments.

### Wav2Vec2 Fine-Tuning
- **Base Model**: [Wav2Vec2ForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/wav2vec2).
- **Fine-Tuning**: We load a pretrained checkpoint and adapt the final classification head for 3 classes.
- **Implementation**: In `src/training_wave2vec2.py`, we use the Hugging Face Trainer or a custom training loop, specifying the same labeled data.

> **Note**: Both trainings produce their own best model checkpoints:  
> - `models/yamnet_finetuned.h5`  
> - `models/wav2vec2_finetuned/` (directory with PyTorch checkpoints)

---

## Ensemble Model Development

### Ensemble Techniques
The assignment recommends:
- **Averaging Probabilities**
- **Majority Voting**
- **Meta-classifier** (e.g., training a small network on top of the outputs)

### Final Ensemble Approach
We chose **probability averaging** as it was straightforward and yielded strong performance:
1. Obtain softmax probabilities from YAMNet.
2. Obtain softmax probabilities from Wav2Vec2.
3. **Average** them element-wise:  
   \[
   P_\text{combined} = \frac{P_\text{YAMNet} + P_\text{Wav2Vec2}}{2}
   \]
4. Classify using `argmax(P_combined)`.

Implementation can be found in `src/ensemblemodel.py`.

---

## Training, Testing, and Validation Approach

### Training Split
- **70%** for training
- **15%** for validation
- **15%** for testing
- Ensured **stratified** distribution across cry, scream, normal classes.

### Validation & Early Stopping
- We monitor **validation loss** and **accuracy**.
- Use **early stopping** to avoid overfitting.

### Testing
- Final models are tested on the **unseen 15%** test set.
- We compute **accuracy, precision, recall, F1-score** for each class.
- Additionally, we generate **confusion matrices** and **ROC curves**.

---

## Performance Metrics & Results

### Confusion Matrices & Classification Reports
- See [`plots/ensemble_classification_report.pdf`](./plots/ensemble_classification_report.pdf) for a summary of the ensemble classification metrics and confusion matrix.
- For individual models, see:
  - [`plots/classification_report_yamnet.pdf`](./plots/classification_report_yamnet.pdf)
  - [`plots/ensemble_roc_curves.pdf`](./plots/ensemble_roc_curves.pdf) for ROC curves of the ensemble approach.

### ROC Curves
- The ensemble ROC curves are in [`plots/ensemble_roc_curves.pdf`](./plots/ensemble_roc_curves.pdf). Each class (cry, scream, normal) is plotted separately.

> **Overall**, the ensemble model typically outperforms individual models, showing higher recall for cry and scream classes and balanced precision for normal utterances.

---

## Loss Function Justification
For **multi-class classification** (cry, scream, normal), we use **categorical cross-entropy** (in Keras) or **cross-entropy loss** (in PyTorch). This is suitable because:
1. **Probabilistic Outputs**: We want each model to output probabilities over the 3 classes.
2. **One-hot Targets**: Each audio segment belongs to exactly one class.
3. **Experimental vs. Control**: The data has two “experimental” classes (cry, scream) and one “control” class (normal speech/music/other). Cross-entropy is standard for such tasks.

---

## Deployment with Temporal

### Temporal Workflow Design
The assignment requires designing a Temporal workflow to:
1. **Receive and preprocess audio input** (5–15 seconds each).
2. **Run the ensemble model** for classification.
3. **Store and manage results** (e.g., logs, metrics).

**Workflow Steps**:
- `workflow.py`: Defines the workflow structure (what tasks to call in sequence).
- `worker.py`: Defines the worker that executes the activities (preprocessing, inference).
- `client.py`: Submits workflow executions.

### Real-time Audio Handling
- The assignment states the workflow **should handle real-time audio streams** of **at least 5 seconds** and up to **15 seconds**. In practice, you can:
  - Stream or chunk audio to the workflow.
  - Each chunk triggers an activity that runs the ensemble model.

---

## Setup & Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/username/my_project.git
   cd my_project

---

## Data Acquisition & Preprocessing

### Dataset Selection
We use several open-source datasets for:
- **Infant Cry**: [AudioSet 1](https://research.google.com/audioset/dataset/baby_cry_infant_cry.html), [AudioSet 2](https://research.google.com/audioset/dataset/crying_sobbing.html)
- **Screams**: [AudioSet 3](https://research.google.com/audioset/dataset/screaming.html?utm_source=chatgpt.com)
- **Normal Utterances**: [Common Voice Dataset](https://commonvoice.mozilla.org/)

### Data Preparation
1. **Consistent Sampling Rate**: We resample all audio to 16kHz.
2. **Segmentation**: Each audio file is segmented into short clips (5-15 seconds).
3. **Labeling**: Segments are labeled as `crying`, `screaming`, or `normal`.
4. **Metadata**: A CSV (`data_manifest.csv`) keeps track of `file_path` and `label`.

### Sample Size Justification
- We aimed for a balanced dataset with enough examples of each class (cry, scream, normal speech). 
- **Experimental**: cry and scream.  
- **Control**: normal speech, music, ambient indoor sounds.
- To ensure each class is well-represented, we used **stratified sampling** to select training/validation/test splits. This helps maintain class distribution in each split.

---

## Model Training

### YAMNet Fine-Tuning
- **Base Model**: [YAMNet](https://tfhub.dev/google/yamnet/1) which is pretrained on AudioSet.
- **Fine-Tuning**: We replace the classification layer to output 3 classes and freeze/unfreeze select layers.
- **Implementation**: In `src/training.py` (or a separate script), we load YAMNet from TF Hub, adapt final layers, and train on our labeled segments.

### Wav2Vec2 Fine-Tuning
- **Base Model**: [Wav2Vec2ForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/wav2vec2).
- **Fine-Tuning**: We load a pretrained checkpoint and adapt the final classification head for 3 classes.
- **Implementation**: In `src/training_wave2vec2.py`, we use the Hugging Face Trainer or a custom training loop, specifying the same labeled data.

> **Note**: Both trainings produce their own best model checkpoints:  
> - `models/yamnet_finetuned.h5`  
> - `models/wav2vec2_finetuned/` (directory with PyTorch checkpoints)

---

## Ensemble Model Development

### Ensemble Techniques
The assignment recommends:
- **Averaging Probabilities**
- **Majority Voting**
- **Meta-classifier** (e.g., training a small network on top of the outputs)

### Final Ensemble Approach
We chose **probability averaging** as it was straightforward and yielded strong performance:
1. Obtain softmax probabilities from YAMNet.
2. Obtain softmax probabilities from Wav2Vec2.
3. **Average** them element-wise:  
   \[
   P_\text{combined} = \frac{P_\text{YAMNet} + P_\text{Wav2Vec2}}{2}
   \]
4. Classify using `argmax(P_combined)`.

Implementation can be found in `src/ensemblemodel.py`.

---

## Training, Testing, and Validation Approach

### Training Split
- **70%** for training
- **15%** for validation
- **15%** for testing
- Ensured **stratified** distribution across cry, scream, normal classes.

### Validation & Early Stopping
- We monitor **validation loss** and **accuracy**.
- Use **early stopping** to avoid overfitting.

### Testing
- Final models are tested on the **unseen 15%** test set.
- We compute **accuracy, precision, recall, F1-score** for each class.
- Additionally, we generate **confusion matrices** and **ROC curves**.

---

## Performance Metrics & Results

### Confusion Matrices & Classification Reports
- See [`plots/ensemble_classification_report.pdf`](./plots/ensemble_classification_report.pdf) for a summary of the ensemble classification metrics and confusion matrix.
- For individual models, see:
  - [`plots/classification_report_yamnet.pdf`](./plots/classification_report_yamnet.pdf)
  - [`plots/ensemble_roc_curves.pdf`](./plots/ensemble_roc_curves.pdf) for ROC curves of the ensemble approach.

### ROC Curves
- The ensemble ROC curves are in [`plots/ensemble_roc_curves.pdf`](./plots/ensemble_roc_curves.pdf). Each class (cry, scream, normal) is plotted separately.

> **Overall**, the ensemble model typically outperforms individual models, showing higher recall for cry and scream classes and balanced precision for normal utterances.

---

## Loss Function Justification
For **multi-class classification** (cry, scream, normal), we use **categorical cross-entropy** (in Keras) or **cross-entropy loss** (in PyTorch). This is suitable because:
1. **Probabilistic Outputs**: We want each model to output probabilities over the 3 classes.
2. **One-hot Targets**: Each audio segment belongs to exactly one class.
3. **Experimental vs. Control**: The data has two “experimental” classes (cry, scream) and one “control” class (normal speech/music/other). Cross-entropy is standard for such tasks.

---

## Deployment with Temporal

### Temporal Workflow Design
The assignment requires designing a Temporal workflow to:
1. **Receive and preprocess audio input** (5–15 seconds each).
2. **Run the ensemble model** for classification.
3. **Store and manage results** (e.g., logs, metrics).

**Workflow Steps**:
- `workflow.py`: Defines the workflow structure (what tasks to call in sequence).
- `worker.py`: Defines the worker that executes the activities (preprocessing, inference).
- `client.py`: Submits workflow executions.

### Real-time Audio Handling
- The assignment states the workflow **should handle real-time audio streams** of **at least 5 seconds** and up to **15 seconds**. In practice, you can:
  - Stream or chunk audio to the workflow.
  - Each chunk triggers an activity that runs the ensemble model.

---