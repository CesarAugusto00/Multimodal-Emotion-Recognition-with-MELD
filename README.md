# Multimodal Emotion Recognition with MELD

### Text–Audio Fusion Inspired by DialogueTMR

## Overview

This project explores **multimodal emotion recognition** using the MELD dataset, combining textual and acoustic information to predict one of the seven emotion categories defined in MELD.

The project was developed as a **three-day machine learning prototype**. Given this time constraint, the objective was not to fully reproduce a large state-of-the-art architecture, but rather to design, train, evaluate, and demonstrate a complete multimodal emotion-recognition pipeline.

To make experimentation feasible within the available time, the system was built around a relatively lightweight architecture. This allowed the project to focus on the interaction between pretrained text and audio representations, modality-specific transformations, multimodal fusion, and emotion classification while maintaining manageable training times.

The architecture is inspired by ideas from **DialogueTMR**, particularly the use of modality-specific representations and cross-modal interaction, while simplifying the overall design to suit the scope and computational constraints of the project.

---

## Motivation

Emotion cannot always be reliably inferred from words alone.

The same sentence may communicate very different emotional states depending on **vocal tone, emphasis, timing, intensity, and prosody**. Text provides strong semantic information about what is being said, while audio provides complementary information about how it is being said.

Combining these two modalities can therefore provide a richer representation of the speaker's emotional state than either modality independently.

The **MELD (Multimodal EmotionLines Dataset)** provides conversational text, audio, video, speaker information, and emotion annotations, making it well suited for exploring multimodal emotion recognition.

This project focuses specifically on the **Text + Audio** modalities.

---

## Project Documentation

The project is divided into several stages. More detailed documentation about the development process, dataset, model architecture, training procedure, experimental results, and future extensions is available in the following sections:

### [Project Timeline](docs/PROJECT_TIMELINE.md)

A chronological overview of the three-day development process, from the initial state-of-the-art review to the final architecture. This document describes the approaches explored, major design decisions, experiments, limitations, and changes made throughout development.

### [Dataset and Preprocessing](docs/DATASET.md)

Detailed information about the MELD dataset and how it was prepared for this project, including the dataset structure, emotion classes, text normalization, audio preprocessing, class distribution, dialogue-level train/validation split, and feature extraction using DistilBERT and Wav2Vec2.

### [Stage 1 — Intra-Modal Transformation](docs/INTRA_MODAL_TRANSFORMATION.md)

Description of the first stage of the emotion-recognition model, where pretrained text and audio representations are independently transformed while incorporating conversational context. This section covers the rolling context window, Transformer architecture, training procedure, implementation, and experimental results.

### [Stage 2 — Multi-Grain Interactive Fusion](docs/MULTIGRAIN_FUSION.md)

Description of the multimodal fusion stage inspired by DialogueTRM, where the contextualized text and audio representations interact before being passed to the emotion classifier. This section covers the fusion strategy, MGIF-inspired architecture, training procedure, comparison with the concatenation baseline, and evaluation results.

### [Results and Complete Pipeline](docs/RESULTS_AND_PIPELINE.md)

Presentation of the experimental results and the complete end-to-end inference architecture. This document describes how raw text and audio are processed through DistilBERT, Wav2Vec2, the intra-modal Transformers, multimodal fusion, and emotion classifier. It also explains how the resulting structured emotional state is connected to **Llama 3.2 1B through Ollama** to generate a short emotion-aware response.

### [Future Work](docs/FUTURE_WORK.md)

Possible extensions toward a fully real-time emotion-aware conversational system. These include microphone input, Voice Activity Detection (VAD), automatic utterance segmentation, speech-to-text transcription, automatic dialogue/session management, text-to-speech output, streaming inference, and latency optimization.

The proposed future architecture extends the current prototype toward the following real-time interaction loop:

```text
Microphone
    │
    ▼
Voice Activity Detection
    │
    ▼
Utterance Recording
    │
    ├──────────────► Speech-to-Text ──► DistilBERT
    │
    └──────────────► Wav2Vec2
                              │
                              ▼
                   Emotion Recognition
                              │
                              ▼
                   Structured Emotion State
                              │
                              ▼
                       Llama 3.2 1B
                          (Ollama)
                              │
                              ▼
                        Text-to-Speech
                              │
                              ▼
                         Robot Response
```

### [Setup and Running Instructions](code/README.md)

Instructions for setting up and running the complete prototype locally. This guide covers creating a Python virtual environment, installing the required dependencies, setting up **Ollama and Llama 3.2 1B**, preparing the model files, and running the end-to-end pipeline from raw text and audio to emotion classification and response generation.

---

## Dataset

This project uses the **MELD (Multimodal EmotionLines Dataset)**, a conversational multimodal dataset containing utterances extracted from the television series *Friends*.

Each utterance includes multimodal information and an emotion annotation corresponding to one of seven categories:

* Anger
* Disgust
* Fear
* Joy
* Neutral
* Sadness
* Surprise

For this project, only the **textual and acoustic modalities** are used.

A detailed description of the dataset, preprocessing pipeline, feature extraction process, and data distribution can be found in the **[Dataset and Preprocessing documentation](docs/DATASET.md)**.

---

## Architecture

The proposed system follows a two-stage multimodal architecture.

### Stage 1 — Intra-Modal Transformation

Text and audio are first processed independently.

Pretrained language and speech models are used to obtain initial representations for each modality. These embeddings are then passed through modality-specific transformation modules to produce representations better suited for the downstream emotion-recognition task.

```text
Text  ──► Text Encoder  ──► Text Embeddings  ──► Intra-Modal Transformation ──┐
                                                                              │
                                                                              ▼
                                                                      Multimodal Fusion
                                                                              ▲
                                                                              │
Audio ──► Audio Encoder ──► Audio Embeddings ──► Intra-Modal Transformation ──┘
```

More information about this stage can be found in **[Stage 1 — Intra-Modal Transformation](docs/INTRA_MODAL_TRANSFORMATION.md)**.

### Stage 2 — Multi-Grain Interactive Fusion

The transformed text and audio representations are combined using a **multi-grain interactive fusion strategy** designed to capture complementary information across both modalities.

The resulting multimodal representation is then processed by a **Transformer-based classifier** to predict the final emotion category.

```text
Transformed Text Features ──┐
                            │
                            ▼
                  Multi-Grain Interactive
                          Fusion
                            ▲
                            │
Transformed Audio Features ─┘
                            │
                            ▼
                  Transformer Classifier
                            │
                            ▼
                  Emotion Classification
                            │
                            ▼
           Anger | Disgust | Fear | Joy
          Neutral | Sadness | Surprise
```

A detailed explanation of the fusion mechanism and classifier is available in **[Stage 2 — Multi-Grain Interactive Fusion](docs/MULTIGRAIN_FUSION.md)**.

---

## Future Work

Due to the three-day development constraint, the current system should be considered a **functional prototype rather than a fully optimized architecture**.

Several extensions could be explored in future work:

* Perform more extensive hyperparameter optimization.
* Investigate deeper or more expressive modality-specific transformation modules.
* Experiment with alternative text and speech encoders.
* Explore more sophisticated cross-modal attention and fusion mechanisms.
* Address the class imbalance present in MELD more extensively.
* Incorporate the visual modality available in MELD.
* Evaluate the contribution of each modality through ablation experiments.
* Compare the multimodal architecture against stronger unimodal and multimodal baselines.
* Explore conversational context across multiple utterances rather than classifying each utterance independently.
* Optimize the complete pipeline for real-time inference.

These extensions would help determine how the prototype architecture scales beyond the computational and time constraints of the initial project.
