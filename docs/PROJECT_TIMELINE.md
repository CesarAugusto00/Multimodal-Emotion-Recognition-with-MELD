# Project Timeline and Design Decisions

This document describes the development process of the multimodal emotion recognition system, from the initial state-of-the-art review to the design and implementation of the final architecture.

The project was developed within a **three-day time constraint**. Therefore, the main objective was to design, train, and evaluate a complete multimodal pipeline while prioritizing architectures that could be realistically implemented with the available time and computational resources.

---

# 0. State-of-the-Art Review

## 0.1 Multimodal Emotion Recognition Survey

The project began with a review of the current state of the art in **Multimodal Emotion Recognition in Conversations (MERC)**.

The main reference used for this initial exploration was:

> **Wu et al. (2025) — *Multimodal Emotion Recognition in Conversations: A Survey of Methods, Trends, Challenges and Prospects***
> **[Read the paper →](../papers/MERC_Survey_2025.pdf)**

The survey provided an overview of recent approaches to multimodal emotion recognition and categorizes current methods into three main groups:

1. **Graph-based methods**
2. **Fusion-based methods**
3. **Generation-based methods**

### Architecture Selection

Given the three-day development constraint, **fusion-based methods** were selected as the main direction for the project.

Rather than implementing the additional conversational structures required by graph-based approaches or relying on a larger generative architecture, the fusion-based approach made it possible to focus directly on the interaction between representations from different modalities.

The next step was therefore to investigate existing fusion-based architectures that could provide a reference for the design of the project.

---

# 1. Exploration of Fusion-Based Models

## 1.1 DialogueTRM

Among the fusion-based models identified during the literature review, **DialogueTRM** was selected for further study.

The architecture is presented in:

> **Mao et al. — *DialogueTRM: Exploring the Intra- and Inter-Modal Emotional Behaviors in the Conversation***
> **[Read the paper →](../papers/DialogueTRM.pdf)**

DialogueTRM was particularly relevant because it explicitly considers two different aspects of multimodal emotion recognition:

* **Intra-modal emotional behavior** — relationships occurring within each individual modality.
* **Inter-modal emotional behavior** — relationships and contributions across different modalities.

The original model operates on three modalities:

```text
Text + Audio + Vision
```

and is organized around three principal components:

```text
Multimodal Inputs
       │
       ▼
Hierarchical Transformer (HT)
       │
       │  Intra-modal modeling
       ▼
Multi-Grained Interactive Fusion (MGIF)
       │
       │  Inter-modal modeling
       ▼
Discriminator
       │
       ▼
Emotion Prediction
```

This architecture became the main conceptual inspiration for the model developed in this project.

---

# 2. Intra-Modal and Inter-Modal Modeling

One of the main ideas adopted from DialogueTRM was the distinction between **intra-modal** and **inter-modal** processing.

## 2.1 Intra-Modal Processing

The intra-modal stage considers the behavior of each modality independently.

DialogueTRM argues that different modalities can have different dependencies on conversational context. Textual information, for example, may depend strongly on previous utterances, while acoustic or visual expressions can contain more immediate emotional information.

In the original DialogueTRM architecture, this behavior is modeled through a **Hierarchical Transformer (HT)**.

For this project, the underlying idea was retained but the architecture was simplified:

```text
Text Embedding
      │
      ▼
Text Intra-Modal Transformation
      │
      ▼
Transformed Text Representation
```

and:

```text
Audio Embedding
      │
      ▼
Audio Intra-Modal Transformation
      │
      ▼
Transformed Audio Representation
```

The objective is to first obtain useful modality-specific representations before attempting to combine information between modalities.

---

## 2.2 Inter-Modal Processing

The inter-modal stage focuses on the interaction **between different modalities**.

Text and audio do not necessarily contribute equally to every emotion prediction. In some utterances, the words themselves may provide the strongest emotional signal, while in others the acoustic characteristics of the speaker may provide additional or even contradictory information.

DialogueTRM addresses this through its **Multi-Grained Interactive Fusion (MGIF)** module.

The MGIF architecture performs multimodal interaction at two levels:

```text
Neuron-Grained Fusion
        │
        ▼
Vector-Grained Fusion
```

The neuron-grained stage uses a multimodal gating mechanism to interactively combine features from different modalities, while the vector-grained stage uses Transformer attention to learn the relative importance of the resulting fused representations.

This concept provided the foundation for the multimodal fusion stage implemented in this project.

---

# 3. Simplifying DialogueTRM

The objective of this project was **not to reproduce DialogueTRM in full**.

Instead, DialogueTRM was used as architectural inspiration for the development of a smaller Text + Audio system that could be implemented and evaluated within the three-day project timeline.

Several simplifications were therefore introduced.

## 3.1 Removing the Visual Modality

The original DialogueTRM architecture operates on:

```text
Text + Audio + Vision
```

For this project, the visual modality was removed, resulting in:

```text
Text + Audio
```

This substantially reduced the preprocessing requirements and computational complexity of the pipeline.

It also allowed the available development time to be concentrated on understanding the interaction between **linguistic and acoustic emotional information**.

---
