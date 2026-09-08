# Results

## Overview

The final emotion-recognition prototype uses **Text + Audio** to predict
one of the seven MELD emotion categories: anger, disgust, fear, joy,
neutral, sadness, and surprise.

The system combines DistilBERT text representations and Wav2Vec2 audio
representations with short conversational context. Separate intra-modal
Transformer modules model recent text and audio sequences, followed by a
DialogueTRM-inspired multi-grain fusion module and an emotion
classifier.

The final evaluation reported here was performed through the
**end-to-end inference pipeline**, starting from utterance text and raw
media files rather than directly loading precomputed embeddings.

## Evaluation Setup

The MELD training data was divided by `Dialogue_ID` into training and
validation subsets. Splitting by dialogue prevents utterances from the
same conversation from appearing in both subsets.

The validation evaluation contained **2,091 utterances**. The pipeline
processed all 2,091 successfully, with **0 failed media files**.
Conversational context was reset whenever a new dialogue began.

The primary metrics were Accuracy, Macro F1, and Weighted F1. Macro F1
is particularly important because MELD is imbalanced and it gives each
emotion class equal importance.

## End-to-End Results

  | Metric | Result |
|---|---:|
| Accuracy | **0.6016** |
| Macro F1 | **0.3587** |
| Weighted F1 | **0.5767** |
| Processed utterances | **2,091** |
| Failed media files | **0** |

The end-to-end system correctly classified approximately **60.2%** of
the validation utterances. Macro F1 is substantially lower than accuracy
because performance varies considerably across emotion classes and the
dataset is imbalanced.



## Per-Class Results

  | Emotion | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Anger | 0.3745 | 0.4159 | 0.3941 | 226 |
| Disgust | 0.0667 | 0.0208 | 0.0317 | 48 |
| Fear | 0.1111 | 0.0233 | 0.0385 | 43 |
| Joy | 0.5880 | 0.4209 | 0.4906 | 373 |
| Neutral | 0.6932 | 0.8640 | 0.7692 | 978 |
| Sadness | 0.3363 | 0.2390 | 0.2794 | 159 |
| Surprise | 0.5622 | 0.4621 | 0.5073 | 264 |

Performance was strongest for **neutral (F1 = 0.7692)**, followed by
**surprise (0.5073)** and **joy (0.4906)**. The most difficult classes
were **disgust (0.0317)** and **fear (0.0385)**. These are also the two
classes with the smallest validation support.

Neutral recall reached **0.8640**, showing that the classifier
recognizes the majority neutral class particularly well. The weaker
minority-class performance explains much of the gap between accuracy and
Macro F1.

## Model Development Comparison

### Concatenation Baseline

The baseline independently contextualized text and audio with the
intra-modal Transformers. The current contextual text and audio
representations were concatenated and passed to a feed-forward
classifier.

Best observed validation results:

| Metric | Result |
|---|---:|
| Accuracy | **0.6251** |
| Macro F1 | **0.3762** |
| Weighted F1 | **0.5917** |




The best Macro F1 was obtained around epoch 21. Performance later
plateaued and became noisy.

### DialogueTRM-Inspired Multi-Grain Fusion

A second approach replaced simple concatenation with a lightweight
**DialogueTRM-inspired multi-grain interactive fusion (MGIF)** module
adapted to the two-modality Text + Audio architecture.

The implementation combines a modality-level scalar weighting mechanism,
a feature-level gate, and a projection that combines the two fused
representations before classification.

This is a simplified adaptation inspired by DialogueTRM rather than a
reproduction of the complete DialogueTRM architecture.


  | Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Concatenation baseline | **0.6251** | **0.3762** | **0.5917** |
| MGIF, intra-modal Transformers frozen | 0.5973 | 0.3712 | 0.5772 |
| MGIF, fine-tuned | 0.5940 | 0.3737 | 0.5749 |

Fine-tuning improved MGIF Macro F1 from **0.3712 to 0.3737**, but the
simpler concatenation baseline remained the strongest model according to
validation Macro F1.

This is an important experimental result: increasing fusion complexity
did not automatically improve classification performance.

## End-to-End Pipeline Verification

The fine-tuned MGIF pipeline was additionally evaluated by
reconstructing the complete inference path from the original inputs.


  | Metric | Fine-tuned MGIF validation | Raw end-to-end pipeline |
|---|---:|---:|
| Accuracy | 0.5940 | **0.6016** |
| Macro F1 | **0.3737** | 0.3587 |
| Weighted F1 | 0.5749 | **0.5767** |

The raw end-to-end results are broadly consistent with the
development-time validation measurements. Weighted F1 differs by less
than 0.002, accuracy increased slightly, and Macro F1 decreased by
approximately 0.015.

This verifies that the deployed pipeline can start from **raw utterance
text and media**, independently create the DistilBERT and Wav2Vec2
representations, maintain conversational context, execute multimodal
fusion, and return an emotion prediction.

## Key Findings

1.  **Text + Audio emotion recognition works end-to-end.** All 2,091
    validation utterances were processed with no failed media files.
2.  **Conversational context is incorporated into inference.** Recent
    text and audio representations are maintained within each dialogue
    and reset at dialogue boundaries.
3.  **The simpler fusion baseline performed best.** Concatenation
    achieved the highest observed validation Macro F1 of **0.3762**.
4.  **MGIF did not outperform concatenation.** Fine-tuning helped
    slightly relative to frozen MGIF, but not enough to beat the
    baseline.
5.  **Class imbalance is a major limitation.** Neutral performs
    substantially better than fear and disgust.
6.  **Accuracy alone is insufficient.** End-to-end accuracy was
    **0.6016**, compared with Macro F1 of **0.3587**.
7.  **Raw-input inference is consistent with development-time
    evaluation.** Similar accuracy and Weighted F1 support that
    preprocessing and model inference are connected correctly.

## Limitations

The main limitation is performance on low-frequency emotion classes.
Fear and disgust have very few validation examples and correspondingly
low recall.

The prototype uses a fixed short conversational context rather than
arbitrarily long dialogue memory. It assumes one active dialogue and
explicitly resets context when a new dialogue begins.

The reported metrics are **validation results** from a dialogue-level
split of the available MELD training data. They should not be described
as performance on an untouched official test set.

The fusion experiments were also constrained by the take-home challenge
timebox. The goal was a focused, understandable end-to-end prototype
rather than exhaustive architecture optimization.

## Potential Improvements

Future work could investigate:

-   class-weighted cross-entropy or focal loss;
-   balanced or weighted sampling;
-   audio augmentation for minority classes;
-   additional hyperparameter tuning;
-   calibrated emotion confidence;
-   longer or adaptive conversational context;
-   explicit uncertainty handling;
-   adding the visual modality;
-   adaptive real-time speech endpoint detection;
-   speaker identification for multi-speaker conversations.

## Response Generation

The emotion-recognition pipeline is designed to pass a structured
prediction to a lightweight local response generator. For example:

``` python
{
    "emotion": "sadness",
    "confidence": 0.82,
    "text": "I had a really difficult day."
}
```

The response-generation stage can combine this emotional state with the
current utterance and a short history of previous user/robot exchanges.
A small local language model can then generate a short conversational
response.

This separation keeps the architecture interpretable: the multimodal
model performs emotion recognition, while the language model performs
response generation.

The response-generation component is separate from the quantitative
classification results reported above. Its final integration and latency
should be added once the Llama-based response pipeline is complete.
