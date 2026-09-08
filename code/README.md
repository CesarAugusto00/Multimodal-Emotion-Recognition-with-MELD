# Multimodal Emotion Recognition with MELD

This project implements a multimodal conversational emotion recognition pipeline using **text + audio**.

The system takes a user utterance and an audio file, estimates the emotion of the current utterance, and then generates a short conversational response using **Llama 3.2 1B through Ollama**.

## Pipeline

```text
Text + Audio
    |
    +--> DistilBERT text encoder
    |
    +--> Wav2Vec2 audio encoder
              |
              v
     Intra-modal Transformers
              |
              v
      Multimodal Fusion (MGIF)
              |
              v
       Emotion Classification
              |
              v
     Emotion + Confidence + Text
              |
              v
         Llama 3.2 1B
              |
              v
        Robot Response
```

The emotion classifier predicts one of the seven MELD emotion classes:

- anger
- disgust
- fear
- joy
- neutral
- sadness
- surprise

---

## Project Structure

```text
PPP/
│
├── encoders/
│   ├── __init__.py
│   ├── text_encoder.py
│   └── audio_encoder.py
│
├── models/
│   ├── __init__.py
│   ├── intramodal.py
│   ├── mgif.py
│   └── emotion_model.py
│
├── pipeline/
│   ├── __init__.py
│   ├── emotion_pipeline.py
│   └── response_generator.py
│
├── Model/
│   └── mgif_emotion_model.pt
│
├── test_data/
│   └── example.wav
│
├── testing.py
└── requirements.txt
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/CesarAugusto00/Multimodal-Emotion-Recognition-with-MELD.git
cd Multimodal-Emotion-Recognition-with-MELD
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv winvenv
```

Activate it with:

```powershell
.\winvenv\Scripts\Activate.ps1
```

#### Linux / WSL

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install the project requirements

```bash
pip install -r requirements.txt
```

The project requires packages including:

```text
torch
transformers
librosa
numpy
scikit-learn
soundfile
ollama
```

---

## Ollama Setup

The response-generation stage uses **Llama 3.2 1B** through Ollama.

Install Ollama from the official Ollama website, then download the model:

```bash
ollama pull llama3.2:1b
```

Verify that the model is available:

```bash
ollama list
```

Test the model directly with:

```bash
ollama run llama3.2:1b
```

---

## Important Note for WSL Users

If Ollama is installed on Windows while Python is running inside WSL, WSL may not be able to reach the Windows Ollama server through `localhost`.

For the simplest setup, run the project entirely using a **Windows Python virtual environment** while Ollama is also running on Windows.

```text
Windows
│
├── Ollama
│   └── llama3.2:1b
│
└── Python
    └── winvenv
        └── Project
```

---

## Model Files

The trained emotion model checkpoint should be placed at:

```text
Model/mgif_emotion_model.pt
```

The pipeline loads this checkpoint during inference.

The pretrained encoders are downloaded automatically through Hugging Face:

```text
distilbert-base-uncased
facebook/wav2vec2-base
```

An internet connection may therefore be required the first time the program is run.

---

## Running the End-to-End Test

A simple test script is provided in:

```text
testing.py
```

The test sends:

1. User text
2. Corresponding audio
3. Text and audio through the emotion-recognition model
4. Predicted emotion and confidence to Llama
5. Llama generates the robot response

Run:

```bash
python testing.py
```

Example final output:

```python
{
    "emotion": "joy",
    "confidence": 0.78,
    "response": "I'm happy to see you too! It's nice to spend some time together."
}
```

---

## Response Generation

The response generator is implemented in:

```text
pipeline/response_generator.py
```

It uses:

```python
ollama.chat(
    model="llama3.2:1b",
    messages=[...]
)
```

The prompt provides Llama with:

- the current user utterance
- estimated emotion
- emotion confidence
- recent conversation history

The detected emotion is treated as an additional signal rather than absolute ground truth.

---

## Conversational Context

The emotion-recognition pipeline uses a rolling context of up to three user utterances.

The Llama response generator separately uses the most recent user/robot exchanges to maintain conversational continuity.

When starting a new conversation, reset the emotion context:

```python
emotion_pipeline.reset_context()
```

---

## Testing Ollama Separately

Create a small test file:

```python
import ollama

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": "Say hello in one short sentence."
        }
    ]
)

print(response.message.content)
```

Run:

```bash
python test_ollama.py
```

---

## Testing PyTorch

Check that PyTorch is installed:

```bash
python -c "import torch; print(torch.__version__)"
```

Check whether CUDA is available:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Results

The end-to-end multimodal pipeline was evaluated on a dialogue-separated validation split from MELD.

| Metric | Result |
|---|---:|
| Accuracy | **0.6016** |
| Macro F1 | **0.3587** |
| Weighted F1 | **0.5767** |
| Processed utterances | **2,091** |
| Failed media files | **0** |

More detailed results are available in:

```text
docs/RESULTS.md
```

---

## Current Limitations

- Performance is significantly weaker for minority emotions such as fear and disgust.
- The conversation context uses a fixed short window.
- The current system uses text and audio only.
- Emotion confidence is based on classifier softmax output and is not explicitly calibrated.
- Audio decoding and model inference can introduce latency.
- The current evaluation uses a dialogue-separated validation split rather than the official MELD test set.

---

## Potential Improvements

- class-weighted loss or focal loss
- balanced training sampling
- audio augmentation
- confidence calibration
- adaptive conversational context
- visual modality integration
- speaker identification
- voice activity detection
- streaming speech input
- text-to-speech output
- latency optimization
- uncertainty-aware response generation

---

## Git Ignore

Virtual environments should not be uploaded to GitHub.

Add:

```text
venv/
winvenv/
__pycache__/
*.pyc
```

Large model files may also need to be excluded or distributed separately depending on GitHub file-size limits.
