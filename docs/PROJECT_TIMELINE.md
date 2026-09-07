# Project Timeline and Design Decisions

This document describes the development process of the multimodal emotion recognition system, from the initial state-of-the-art review to the design and implementation of the final architecture.

The project was developed within a **three-day time constraint**. Therefore, the main objective was to design, train, and evaluate a complete multimodal pipeline while prioritizing architectures that could be realistically implemented with the available time and computational resources.

---

# 0. State-of-the-Art Review

## 0.1 Multimodal Emotion Recognition Survey

The project began with a review of the current state of the art in **Multimodal Emotion Recognition in Conversations (MERC)**.

The main reference used for this initial exploration was:

> **Wu et al. (2025) — *Multimodal Emotion Recognition in Conversations: A Survey of Methods, Trends, Challenges and Prospects***
[Read the MERC Survey →](../papers/MERC_Survey_2025.pdf)

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

# 4. Dataset Preparation and Cleaning

After selecting the Text + Audio architecture, the next stage consisted of preparing the MELD data for model development.

The preprocessing stage included:

- Loading the MELD training metadata.
- Checking for missing values.
- Normalizing text encoding and punctuation.
- Associating each utterance with its corresponding media file.
- Identifying unusable media.
- Extracting modality-specific representations.
- Constructing dialogue-level contextual samples.

The processed dataset contained no missing values in the final dataframe.

Some utterances contained incorrectly represented punctuation or special characters. These were normalized using a Latin-1 to CP1252 conversion, replacement of common typographic punctuation with ASCII equivalents, and Unicode NFKD normalization.

One media example could not be decoded correctly and was removed so that the remaining audio preprocessing could continue.

After preprocessing, the MELD training data contained **9,988 usable utterances** distributed across the seven target emotion classes.

A more detailed description of the dataset, cleaning procedure, class distribution, media preparation, and preprocessing decisions is available here:

[Read the Dataset Preparation Documentation →](DATASET.md)


---

# 5. End-to-End Inference Pipeline

Once the modality encoders, intra-modal Transformers, and multimodal fusion model had been trained, the next objective was to connect the individual components into a complete inference pipeline.

During model development, text and audio embeddings were precomputed to reduce repeated computation. However, a deployed system must be capable of processing raw inputs.

The inference pipeline therefore performs feature extraction directly from the incoming text and audio.

The resulting architecture is:

```text
                    Raw Utterance
                    /           \
                   /             \
                  ▼               ▼
          Utterance Text      Audio Signal
                  │               │
                  ▼               ▼
            DistilBERT         Wav2Vec2
                  │               │
                  ▼               ▼
          Text Embedding     Audio Embedding
                  │               │
                  └──────┬────────┘
                         │
                         ▼
                Rolling Context Buffer
                  (3 utterances)
                         │
                 ┌───────┴────────┐
                 ▼                ▼
        Text Intra-Modal   Audio Intra-Modal
          Transformer        Transformer
                 │                │
                 └───────┬────────┘
                         ▼
              Multi-Grained Fusion
                     (MGIF)
                         │
                         ▼
                Emotion Classifier
                         │
                         ▼
          Emotion + Confidence Score
```

The text encoder uses:

```text
distilbert-base-uncased
```

while the audio encoder uses:

```text
facebook/wav2vec2-base
```

The pipeline maintains a rolling conversational context containing up to three utterances:

```text
U1: [PAD, PAD, U1]
U2: [PAD, U1,  U2]
U3: [U1,  U2,  U3]
U4: [U2,  U3,  U4]
```

The context is reset whenever a new conversation begins.

This stage also serves an important deployment purpose: it verifies that the preprocessing performed during training can be reproduced from raw input at inference time.

The main implementation is available here:

[View the Emotion Pipeline →](../pipeline/emotion_pipeline.py)

---

# 6. Full-Pipeline Validation

After integrating the individual components, the complete pipeline is evaluated using raw MELD utterances and their corresponding media files.

The official MELD test media downloaded for this project could not be decoded reliably. The exact cause was not established and may be related to media formatting, extraction, codec compatibility, or file integrity.

Therefore, evaluation during development uses a held-out validation portion created from the successfully processed MELD training data.

The split was performed at the **dialogue level** rather than at the individual utterance level:

```python
from sklearn.model_selection import train_test_split

dialogue_ids = df["Dialogue_ID"].unique()

train_ids, val_ids = train_test_split(
    dialogue_ids,
    test_size=0.2,
    random_state=42
)
```

This results in approximately:

```text
80% of dialogues → Training
20% of dialogues → Validation
```

Splitting by `Dialogue_ID` ensures that complete conversations remain within a single subset.

This is particularly important because the proposed architecture uses previous utterances as contextual information. Randomly splitting individual utterances could allow neighboring utterances from the same conversation to appear in both training and validation data, potentially introducing information leakage.

For full-pipeline validation, each dialogue is processed in chronological order:

```text
MELD Media + Transcript
          │
          ▼
Raw-Input Encoders
          │
          ▼
Context Buffer
          │
          ▼
Intra-Modal Transformers
          │
          ▼
MGIF
          │
          ▼
Predicted Emotion
          │
          ▼
Comparison with MELD Label
```

This evaluation is different from evaluating previously stored embeddings.

Its purpose is to verify the complete deployment path:

1. Media decoding.
2. Audio preprocessing.
3. Text encoding.
4. Audio encoding.
5. Context-buffer construction.
6. Intra-modal transformation.
7. Multimodal fusion.
8. Emotion prediction.

The results from this stage should therefore be interpreted as **validation performance**, rather than performance on the official MELD test split.

---

# 7. Local Response Generation

Emotion recognition represents the first output of the system.

The final stage of the prototype converts the detected multimodal emotional state into a short, context-aware response suitable for a conversational robot.

To keep the complete system lightweight and capable of local inference, the selected language model is:

**Meta Llama 3.2 1B Instruct**

The model is executed locally using **Ollama**.

The multimodal model and the language model have separate responsibilities:

```text
Multimodal Model
       │
       │ Determines emotional state
       ▼
Structured Emotional State
       │
       │ Conditions response generation
       ▼
Llama 3.2 1B
       │
       ▼
Natural-Language Response
```

The LLM does not receive raw audio.

Instead, the multimodal emotion-recognition pipeline first produces a structured state containing information such as:

```json
{
    "emotion": "sadness",
    "confidence": 0.72,
    "utterance": "I really thought things would work out."
}
```

This information is then passed to Llama together with limited conversational context.

The complete architecture becomes:

```text
                       User
                        │
                 Text + Audio
                        │
                        ▼
              Multimodal Pipeline
                        │
                        ▼
               Structured State
         ┌─────────────────────────┐
         │ emotion: "sadness"      │
         │ confidence: 0.72        │
         │ utterance: "..."        │
         └─────────────────────────┘
                        │
                        ▼
                 Llama 3.2 1B
                 through Ollama
                        │
                        ▼
              Short Emotion-Aware
                  Robot Response
```

An example prompt structure is:

```text
You are the response component of an emotion-aware conversational robot.

Detected emotion: sadness
User utterance: "I really thought things would work out."

Respond naturally and briefly.
Take the detected emotional state into account without explicitly telling the
user that an emotion classifier was used.
```

Ollama is used to execute the language model locally rather than relying on a remote inference API.

The response-generation component is intentionally separated from the emotion classifier. The multimodal model is responsible for recognizing the emotional state, while the LLM is responsible for generating the final natural-language response.

This modular design also makes it possible to replace the response-generation model without retraining the multimodal emotion classifier.

---

# 8. Final System Architecture

The complete system can therefore be summarized as:

```text
Raw Text ─────► DistilBERT ─────┐
                                │
Raw Audio ────► Wav2Vec2 ───────┤
                                ▼
                       Contextual Modeling
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
        Text Intra-Modal              Audio Intra-Modal
          Transformer                   Transformer
                 │                             │
                 └──────────────┬──────────────┘
                                ▼
                              MGIF
                                │
                                ▼
                       Emotion Prediction
                                │
                                ▼
                    Structured Emotional State
                                │
                                ▼
                        Llama 3.2 1B
                           (Ollama)
                                │
                                ▼
                       Robot Response
```

The system therefore separates the problem into two primary components:

1. **Multimodal perception** — determine the emotional state from linguistic and acoustic evidence.
2. **Response generation** — generate a short response conditioned on the predicted emotional state.

This separation makes the architecture easier to test, interpret, and modify component-by-component.

---

# 9. Model Parameter Budget

An important design constraint was keeping the complete local inference system below the allowed parameter budget.

The approximate parameter counts of the principal components are:

| Component | Approximate Parameters |
|---|---:|
| DistilBERT Base | ~66–67M |
| Wav2Vec2 Base | ~94–95M |
| Intra-Modal Transformers + MGIF + Classifier | ~1.75M |
| Llama 3.2 1B | ~1.23B |
| **Approximate Total** | **~1.39B** |

Therefore, the complete system contains approximately:

```text
~1.39 billion learned parameters
```

This is substantially below a 6-billion-parameter limit.

The parameter budget can be represented approximately as:

```text
DistilBERT       ~67M   ┐
Wav2Vec2         ~95M   │
Emotion Model    ~1.8M  ├──► ~1.39B total parameters
Llama 3.2 1B   ~1.23B   │
                       ┘
```

The exact parameter counts of the PyTorch components can also be calculated directly:

```python
def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )

print(
    "Emotion model:",
    f"{count_parameters(pipeline.model):,}"
)

print(
    "DistilBERT:",
    f"{count_parameters(pipeline.text_encoder.model):,}"
)

print(
    "Wav2Vec2:",
    f"{count_parameters(pipeline.audio_encoder.model):,}"
)
```

The final documentation should report the exact counts produced by the implemented models when possible.

If Llama is quantized for local inference through Ollama, the quantization reduces the memory and storage requirements of the model but does **not** change the number of learned parameters reported for the architecture.

---

# 10. Final Development Sequence

The overall development process can be summarized as:

```text
State-of-the-Art Review
          │
          ▼
Selection of Fusion-Based Approach
          │
          ▼
DialogueTRM Investigation
          │
          ▼
Simplification to Text + Audio
          │
          ▼
MELD Dataset Preparation
          │
          ▼
DistilBERT + Wav2Vec2 Embeddings
          │
          ▼
Intra-Modal Transformers
          │
          ▼
Baseline Multimodal Fusion
          │
          ▼
MGIF-Inspired Fusion
          │
          ▼
Model Validation
          │
          ▼
Raw-Input Inference Pipeline
          │
          ▼
Full-Pipeline Validation
          │
          ▼
Llama 3.2 1B Response Generation
          │
          ▼
Complete Emotion-Aware
Conversational Prototype
```

The final prototype therefore combines a specialized multimodal emotion-recognition model with a lightweight local language model, allowing emotional perception and natural-language response generation to remain separate but connected components of the same system.

---

# 11. Additional Documentation

The implementation details, experimental results, final inference architecture,
and possible extensions of the prototype are documented separately.

## Results and Complete Pipeline

The final experimental results, selected model, Llama-based response generator,
and connection between multimodal emotion recognition and natural-language
response generation are described here:

[View Results and Complete Pipeline →](RESULTS_AND_PIPELINE.md)

## Future Work

Potential extensions toward a complete real-time conversational robot,
including speech recognition, automatic utterance segmentation, dialogue
management, microphone input, and text-to-speech output, are described here:

[View Future Work →](FUTURE_WORK.md)
