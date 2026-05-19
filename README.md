# Offline Multilingual TTS Terminal Application

A simple, fully offline terminal application that converts text files into speech in multiple languages using Piper TTS.

## Features
- 100% Offline (after initial model download)
- Automatic Language Detection
- Native File Picker
- Supports English, Hindi, Telugu, Tamil, Kannada, and Malayalam
- Saves audio as `outputs/output.wav`

## Prerequisites
- Python 3.8+
- Required libraries (see Installation)

## Installation

1. **Clone or Download** this project to your local machine.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you encounter issues with `piper-tts` on Windows, you may need to install the specific wheel for `onnxruntime` or install it via `pip install onnxruntime` first.*

## Offline Model Setup (MANDATORY)

Since the application runs offline, you must manually download the TTS models for the languages you want to use and place them in the `models/` directory.

### Download Links
For each language, download **both** the `.onnx` and the `.onnx.json` files.

| Language | Model File (.onnx) | Config File (.json) |
| :--- | :--- | :--- |
| **English** | [Download](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx) | [Download](https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json) |
| **Hindi** | [Download](https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/central_hindi_24751/medium/hi_IN-central_hindi_24751-medium.onnx) | [Download](https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/central_hindi_24751/medium/hi_IN-central_hindi_24751-medium.onnx.json) |
| **Telugu** | [Download](https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/vasistha/medium/te_IN-vasistha-medium.onnx) | [Download](https://huggingface.co/rhasspy/piper-voices/resolve/main/te/te_IN/vasistha/medium/te_IN-vasistha-medium.onnx.json) |
| **Tamil** | [Download](https://huggingface.co/rhasspy/piper-voices/resolve/main/ta/ta_IN/kanmani/medium/ta_IN-kanmani-medium.onnx) | [Download](https://huggingface.co/rhasspy/piper-voices/resolve/main/ta/ta_IN/kanmani/medium/ta_IN-kanmani-medium.onnx.json) |
| **Kannada** | [Download](https://huggingface.co/rhasspy/piper-voices/resolve/main/kn/kn_IN/padmini/medium/kn_IN-padmini-medium.onnx) | [Download](https://huggingface.co/rhasspy/piper-voices/resolve/main/kn/kn_IN/padmini/medium/kn_IN-padmini-medium.onnx.json) |
| **Malayalam** | [Download](https://huggingface.co/rhasspy/piper-voices/resolve/main/ml/ml_IN/vanya/medium/ml_IN-vanya-medium.onnx) | [Download](https://huggingface.co/rhasspy/piper-voices/resolve/main/ml/ml_IN/vanya/medium/ml_IN-vanya-medium.onnx.json) |

### Placement
Place all downloaded files into the `models/` folder in the project root:
```
project/
├── models/
│   ├── en_US-lessac-medium.onnx
│   ├── en_US-lessac-medium.onnx.json
│   ├── hi_IN-central_hindi_24751-medium.onnx
│   └── hi_IN-central_hindi_24751-medium.onnx.json
│   └── ...
├── app.py
└── ...
```

## Usage

1. **Run the application**:
   ```bash
   python app.py
   ```
2. **Select a file**: A native file dialog will open. Select a `.txt` file containing the text you want to read.
3. **Wait for synthesis**: The app will detect the language, generate the speech, and save it to `outputs/output.wav`.
4. **Listen**: The audio will play automatically after generation.

## Troubleshooting
- **No Sound**: Ensure your speakers are on and `sounddevice` is working.
- **Language Detection**: For very short snippets, detection might be inaccurate. Ensure your text is long enough for reliable detection.
- **Missing Models**: If you see an error about missing files, double-check that the files in `models/` exactly match the names expected in `app.py`.
