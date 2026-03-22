# ISL Translate: Indian Sign Language Real-Time Translation

## Project Overview
This project is an advanced **Indian Sign Language (ISL) Translation System** designed to bridge the communication gap for the deaf and hard-of-hearing community. It captures ISL gestures in real-time using a webcam, translates them into text using Deep Learning, and then synthesizes the text into spoken language using an AI-based voice assistant.

## Key Features
*   **Real-Time Detection:** Uses MediaPipe for low-latency hand landmark tracking.
*   **Sequential Classification:** Employs a **Bidirectional LSTM (Long Short-Term Memory)** model with an **Attention Mechanism** to accurately recognize temporal patterns in sign gestures.
*   **Voice Integration:** Integrates Microsoft Azure's **Edge-TTS** to provide natural-sounding Indian English voices (Female/Male).
*   **Dynamic UI:** Live webcam overlay showing predicted phrases, confidence levels, and gesture buffer status.

## Technologies Used
*   **Computer Vision:** OpenCV, MediaPipe
*   **Deep Learning:** PyTorch, LSTM, Attention Mechanisms
*   **Data Processing:** NumPy, Pandas, Scikit-learn
*   **Voice Synthesis:** Edge-TTS, Pygame
*   **Visualization:** Matplotlib, Seaborn

## Project Structure
```text
D:\ISL_PROJECT\
├── data/
│   ├── raw/                # Raw .pose and CSV files (iSign v1.1)
│   └── processed/          # Preprocessed .npy and encoder files
├── models/                 # Saved model weights (.pth) and metrics
├── src/
│   ├── preprocess.py       # Data cleaning and sequence building
│   ├── dataset.py          # PyTorch Custom Dataset
│   ├── model.py            # Bidirectional LSTM + Attention model
│   ├── train.py            # Model training pipeline
│   ├── evaluate.py         # Testing and performance reports
│   └── live_detect.py      # Real-time inference and voice output
├── utils/
│   └── keypoint_utils.py   # Keypoint extraction and normalization
├── voice/
│   ├── speaker.py          # TTS synthesis and playback
│   ├── voice_config.py     # Voice personality and settings
│   └── audio_cache/        # Cached MP3 phrases for fast playback
├── requirements.txt        # Project dependencies
└── README.md
```

## Setup and Usage

### 1. Installation
Create a virtual environment and install the required libraries:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# Additional requirements for pose processing
pip install pose-format
```

### 2. Preprocessing
Prepare the raw dataset into balanced sequences for training:
```bash
python src/preprocess.py
```

### 3. Training
Train the LSTM model on your GPU or CPU:
```bash
python src/train.py
```

### 4. Evaluation
Check the accuracy, classification report, and confusion matrix:
```bash
python src/evaluate.py
```

### 5. Live Detection
Start the real-time translator:
```bash
python src/live_detect.py
```

## Model Architecture
The core model is a **Bidirectional LSTM** which processes a sequence of 30 frames (126 keypoints each). The **Attention Mechanism** allows the model to "focus" on the most important frames in a gesture, significantly improving accuracy for complex signs.

## License
MIT License
