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

The project is divided into several stages. More detailed documentation about the development process, dataset, training procedure, and individual model components is available in the following sections:

### [Project Timeline](docs/PROJECT_TIMELINE.md)

A chronological overview of the three-day development process, including the approaches explored, design decisions, experiments, limitations, and changes that led to the final architecture.

### [Dataset and Preprocessing](docs/DATASET.md)

Detailed information about the MELD dataset and how it was prepared for this project, including dataset structure, emotion classes, text and audio preprocessing, class distribution, and feature extraction.

### [Stage 1 — Intra-Modal Transformation](docs/INTRA_MODAL_TRANSFORMATION.md)

Description of the first stage of the model, where pretrained text and audio representations are independently transformed before multimodal interaction. This section covers the architecture, training procedure, implementation, and experimental results.

### [Stage 2 — Multi-Grain Interactive Fusion](docs/MULTIGRAIN_FUSION.md)

Description of the multimodal fusion stage, where the transformed text and audio representations interact before being passed to the final emotion classifier. This section covers the fusion strategy, Transformer-based classification, training procedure, and evaluation results.

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
