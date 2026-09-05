# Multimodal-Emotion-Recognition-with-MELD
Text–Audio Fusion Inspired by DialogueTMR
# Real-Time Multimodal Emotion Recognition with MELD
### Text–Audio Fusion Inspired by DialogueTMR

## Overview

This project explores real-time multimodal emotion recognition using the
MELD dataset. The system combines textual and acoustic information to
predict one of the seven MELD emotion categories.

The project was developed as a time-constrained ML prototype. The main
objective was therefore to build, evaluate, and demonstrate a complete
multimodal inference pipeline rather than reproduce a large
state-of-the-art architecture in full.

The final system uses:

- DistilBERT for textual representations
- Wav2Vec2 for acoustic representations
- A lightweight multimodal fusion module
- A classifier for MELD emotion prediction

A small language model can subsequently consume the predicted emotional
state together with the utterance to generate a short response suitable
for an interactive system.


## Motivation

Emotion cannot always be inferred from words alone.

For example, the same sentence can communicate different emotional states
depending on vocal tone, emphasis, timing, or prosody.

MELD provides aligned conversational text, audio, video, and emotion
annotations, making it suitable for studying multimodal emotion
recognition.

This project focuses on the Text + Audio track.


## From Literature Review to Architecture
## Note add the papers used
Before implementing the model, I reviewed work on multimodal emotion
recognition using MELD.

The literature survey motivated the use of learned multimodal fusion
rather than relying exclusively on simple feature concatenation.

DialogueTMR was particularly useful as architectural inspiration. Its
fusion strategy and contextual modeling illustrate how representations
from different modalities can interact before emotion classification.

Reproducing DialogueTMR was outside the intended scope of this
time-constrained prototype.

Instead, this project investigates a simplified architecture inspired by
those ideas:

Text representation ──┐
                      ├── Multimodal Fusion ── Emotion Classifier
Audio representation ─┘

The goal is to preserve the central idea of learned cross-modal
interaction while keeping the model small, interpretable, and practical
to train and deploy.


## Dataset

MELD (Multimodal EmotionLines Dataset) contains conversational utterances
with:

- text
- audio
- video
- speaker information
- dialogue information
- emotion labels

This implementation uses only text and audio.

The target classes are:

- neutral
- joy
- sadness
- anger
- surprise
- fear
- disgust


## Architecture

### Text Encoder

Each utterance is encoded using DistilBERT:

    Utterance
       ↓
    DistilBERT
       ↓
    768-dimensional representation


### Audio Encoder

The corresponding audio clip is encoded using Wav2Vec2:

    Audio waveform
       ↓
    Wav2Vec2
       ↓
    Temporal representations
       ↓
    Mean pooling
       ↓
    768-dimensional representation


### Multimodal Fusion

The text and audio representations are projected to a common latent
dimension and combined using a lightweight learned fusion mechanism.

    DistilBERT [768] ──> Projection ──┐
                                     │
                                     ├── Fusion ──> Fused representation
                                     │
    Wav2Vec2 [768] ───> Projection ──┘

The fusion module is inspired by multi-grained multimodal fusion
approaches, but intentionally simplified for this prototype.


### Emotion Classifier

The fused representation is passed through a small neural classifier:

    Fused representation
          ↓
        Linear
          ↓
         ReLU
          ↓
       Dropout
          ↓
        Linear
          ↓
    7 emotion classes


## Why Utterance-Level Classification?

MELD contains complete dialogues, and dialogue context can improve emotion
recognition.

However, the core prototype performs classification at the utterance
level.

This was an intentional engineering decision.

An utterance-level architecture:

1. provides a clear multimodal baseline;
2. simplifies real-time inference;
3. reduces training complexity;
4. allows the contribution of audio and text fusion to be measured
   independently from dialogue-context modeling.

Dialogue-level contextual modeling using a Transformer is therefore
considered a possible extension rather than a dependency of the core
system.


## Experiments

The following models are evaluated:

1. Text-only baseline
2. Audio-only baseline
3. Text + Audio concatenation baseline
4. Text + Audio learned fusion

This makes it possible to determine whether multimodal information
improves emotion classification and whether learned fusion provides value
beyond simple concatenation.


## Evaluation

Models are evaluated on the MELD validation/test split using metrics such
as:

- Accuracy
- Weighted F1
- Macro F1
- Per-class precision
- Per-class recall
- Confusion matrix

Inference latency and resource usage are also measured because the target
application is interactive.


## Real-Time Inference

The intended inference pipeline is:

    User speech
        │
        ├── Transcript ──> DistilBERT ──┐
        │                               │
        └── Audio ───────> Wav2Vec2 ────┤
                                        ↓
                                      Fusion
                                        ↓
                                 Emotion prediction
                                        ↓
                                 Structured state
                                        ↓
                              Response generation

Example output:

{
    "emotion": "anger",
    "confidence": 0.81,
    "response": "That sounds frustrating. What happened?"
}


## Response Generation

Emotion recognition and response generation are kept as separate
components.

The multimodal classifier is responsible for estimating emotional state.

A small local language model can then receive:

- the original utterance
- the predicted emotion
- optionally the prediction confidence

and generate a short response grounded in the interaction.

Keeping perception and generation separate makes the architecture easier
to inspect, evaluate, and modify.


## Constraints and Design Decisions

This project was developed under a two-to-three-day time constraint.

As a result, priority was given to:

- completing the full pipeline;
- establishing meaningful baselines;
- evaluating multimodal fusion;
- maintaining a small local inference footprint;
- producing reproducible experiments.

Several potentially valuable extensions were intentionally left outside
the core implementation.


## Future Work

Possible extensions include:

- dialogue-context modeling;
- speaker embeddings;
- Transformer-based contextualization across previous utterances;
- end-to-end fine-tuning of DistilBERT and Wav2Vec2;
- alternative multimodal fusion mechanisms;
- text + audio + vision;
- quantization for lower-latency local inference;
- improved response generation.


## Limitations

The current prototype predicts emotion primarily from the current
utterance.

This means that emotions requiring conversational history may be harder
to identify.

The pretrained text and audio representations are initially frozen,
which reduces compute requirements but prevents the encoders from adapting
specifically to MELD.

MELD is also class-imbalanced, so overall accuracy alone is insufficient
for evaluating model quality.


## Repository Structure

meld-audio-text-emotion/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_embeddings.ipynb
│   ├── 03_audio_embeddings.ipynb
│   └── 04_training_and_evaluation.ipynb
│
├── src/
│   ├── data.py
│   ├── encoders.py
│   ├── fusion.py
│   ├── classifier.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── configs/
│   └── baseline.yaml
│
├── results/
│   ├── metrics.json
│   └── confusion_matrix.png
│
└── demo/
    └── app.py
