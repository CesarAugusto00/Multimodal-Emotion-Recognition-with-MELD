# Multi-Grain Interactive Fusion

This stage extends the intra-modal contextual emotion recognition model with a **Multi-Grain Interactive Fusion (MGIF)** module. The goal is to improve the interaction between the textual and acoustic representations before predicting the emotion of the current utterance.

The implementation uses the previously trained text and audio intra-modal Transformers and introduces a learnable fusion mechanism followed by an emotion classifier.

## Architecture

The pipeline developed in this stage is:

```text
Text Embeddings
      │
      ▼
Text Intra-Modal Transformer
      │
      └────► Contextual Text Representation ────┐
                                                │
                                                ▼
                                               MGIF
                                                │
                                                ▼
                                         Emotion Classifier
                                                │
                                                ▼
                                       MELD Emotion Label
                                                ▲
                                                │
Audio Embeddings                                │
      │                                         │
      ▼                                         │
Audio Intra-Modal Transformer                   │
      │                                         │
      └────► Contextual Audio Representation ───┘
```

The model predicts one of the seven MELD emotion categories:

- Anger
- Disgust
- Fear
- Joy
- Neutral
- Sadness
- Surprise

---

## 1. Loading the Pre-Trained Intra-Modal Transformers

The first step was to reuse the text and audio Transformer weights obtained during the previous intra-modal training stage.

```python
text_transformer.load_state_dict(
    torch.load(
        "text_intra_transformer.pt",
        map_location=device
    )
)

audio_transformer.load_state_dict(
    torch.load(
        "audio_intra_transformer.pt",
        map_location=device
    )
)
```

These Transformers process a rolling conversational context of up to three utterances independently for each modality.

For an utterance \(u_t\), the contextual representations can be written as:

$$
H_T =
\text{Transformer}_T
(T_{t-2}, T_{t-1}, T_t)
$$

and

$$
H_A =
\text{Transformer}_A
(A_{t-2}, A_{t-1}, A_t)
$$

where \(T\) represents text embeddings and \(A\) represents audio embeddings.

Because the current utterance is always located at the final position of the context window, the contextual representations used for fusion are:

$$
h_T = H_T[-1]
$$

$$
h_A = H_A[-1]
$$

Both representations have dimension:

$$
h_T, h_A \in \mathbb{R}^{256}
$$

---

## 2. Multi-Grain Interactive Fusion

Instead of directly concatenating the text and audio representations, the new architecture introduces an **MGIF-inspired fusion layer**.

The implementation combines information at two levels:

1. **Vector-grained fusion**
2. **Neuron-grained fusion**

> **Note:** This implementation is inspired by the multi-grain fusion concept rather than being an exact reproduction of the original DialogueTRM architecture. It was adapted to the two-modality Text + Audio configuration used in this project.

### Vector-Grained Fusion

The vector-grained component learns how much importance should be assigned to each modality as a whole.

A learnable scoring function produces one score for text and one for audio:

$$
s_T = f(h_T)
$$

$$
s_A = f(h_A)
$$

A softmax converts these scores into modality weights:

$$
[\alpha_T, \alpha_A]=\text{softmax}([s_T, s_A])
$$

where:

$$
\alpha_T + \alpha_A = 1
$$

The vector-level representation is then:

$$
h_{\text{vector}}=\alpha_T h_T + \alpha_A h_A
$$

This allows the model to dynamically assign more importance to text or audio depending on the input.

For example, the textual content may be more informative for one utterance, while vocal characteristics may be more informative for another.

### Neuron-Grained Fusion

A single weight for an entire modality may be too coarse. Different dimensions of the learned representations can contain different useful information.

The neuron-grained component therefore learns a 256-dimensional gate:

$$
g =\sigma\left(W_g[h_T;h_A]+b_g\right)
$$

where:

$$
g \in \mathbb{R}^{256}
$$

The representations are combined element-by-element:

$$
h_{\text{neuron}}=g \odot h_T+(1-g)\odot h_A
$$

This gives the model finer control over how information from the two modalities is combined.

---

## 3. Combining Both Fusion Levels

The vector- and neuron-grained representations are concatenated:

$$
h_{\text{combined}}=[h_{\text{vector}};h_{\text{neuron}}]
$$

Since each representation contains 256 dimensions:

$$
h_{\text{combined}}\in\mathbb{R}^{512}
$$

A linear projection reduces this representation back to 256 dimensions:

$$
h_{\text{MGIF}}=W_o h_{\text{combined}}+b_o
$$

The resulting representation is then passed to the emotion classifier.

The implemented MGIF module is:

```python
class MGIF(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()

        self.vector_score = nn.Linear(d_model, 1)
        self.neuron_gate = nn.Linear(d_model * 2, d_model)
        self.output_projection = nn.Linear(d_model * 2, d_model)

    def forward(self, text, audio):
        # Vector-grained fusion
        text_score = self.vector_score(text)
        audio_score = self.vector_score(audio)

        scores = torch.cat([text_score, audio_score], dim=1)
        weights = torch.softmax(scores, dim=1)

        text_weight = weights[:, 0:1]
        audio_weight = weights[:, 1:2]

        vector_fused = (
            text_weight * text +
            audio_weight * audio
        )

        # Neuron-grained fusion
        combined = torch.cat([text, audio], dim=-1)
        gate = torch.sigmoid(self.neuron_gate(combined))

        neuron_fused = (
            gate * text +
            (1 - gate) * audio
        )

        # Combine both fusion levels
        fused = torch.cat(
            [vector_fused, neuron_fused],
            dim=-1
        )

        fused = self.output_projection(fused)

        return fused
```

---

## 4. Emotion Classifier

The fused MGIF representation is passed to a small feed-forward classifier.

```text
MGIF Representation (256)
        │
        ▼
Linear (256 → 128)
        │
        ▼
       ReLU
        │
        ▼
    Dropout (0.2)
        │
        ▼
Linear (128 → 7)
        │
        ▼
Emotion Logits
```

The complete multimodal classifier is:

```python
class MGIFEmotionClassifier(nn.Module):
    def __init__(
        self,
        text_transformer,
        audio_transformer,
        d_model=256,
        num_classes=7
    ):
        super().__init__()

        self.text_transformer = text_transformer
        self.audio_transformer = audio_transformer

        self.mgif = MGIF(d_model=d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, text, audio, padding_mask):

        text_context = self.text_transformer(
            text,
            padding_mask
        )

        audio_context = self.audio_transformer(
            audio,
            padding_mask
        )

        current_text = text_context[:, -1, :]
        current_audio = audio_context[:, -1, :]

        fused = self.mgif(
            current_text,
            current_audio
        )

        logits = self.classifier(fused)

        return logits
```

---

## 5. Two-Stage Training Strategy

Training was performed in two stages.

### Stage 1 — Frozen Intra-Modal Transformers

Initially, the previously trained text and audio Transformers were frozen.

```python
for param in text_transformer.parameters():
    param.requires_grad = False

for param in audio_transformer.parameters():
    param.requires_grad = False
```

Only the newly introduced **MGIF module and classifier** were trained.

The optimizer used was AdamW:

```python
optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=3e-4,
    weight_decay=1e-4
)
```

The best validation result during this stage was approximately:

| Metric | Result |
|---|---:|
| Accuracy | 0.5973 |
| Macro F1 | **0.3712** |
| Weighted F1 | 0.5772 |

The best Macro F1 occurred around **epoch 12**.

Training loss continued decreasing afterward while validation performance stopped improving, suggesting that the new fusion layers were beginning to overfit the training data.

The best frozen checkpoint was therefore preserved as:

```text
best_mgif_frozen.pt
```

---

## 6. Fine-Tuning the Complete Architecture

In the second stage, the best frozen MGIF checkpoint was used as the starting point and the text and audio Transformers were unfrozen.

Different learning rates were used for the previously trained Transformers and the newly introduced fusion/classification layers:

```python
optimizer = torch.optim.AdamW(
    [
        {
            "params": model.text_transformer.parameters(),
            "lr": 2e-5
        },
        {
            "params": model.audio_transformer.parameters(),
            "lr": 2e-5
        },
        {
            "params": model.mgif.parameters(),
            "lr": 1e-4
        },
        {
            "params": model.classifier.parameters(),
            "lr": 1e-4
        },
    ],
    weight_decay=1e-4
)
```

A lower learning rate was intentionally used for the intra-modal Transformers to avoid rapidly modifying the representations learned during the previous training stage.

Gradient clipping was also applied:

```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=1.0
)
```

Early stopping was used to prevent unnecessary training after validation performance stopped improving.

The best fine-tuned model occurred at approximately **epoch 4**:

| Metric | Result |
|---|---:|
| Accuracy | 0.5940 |
| Macro F1 | **0.3737** |
| Weighted F1 | 0.5749 |

The checkpoint was saved as:

```text
best_mgif_finetuned.pt
```

A deployment checkpoint containing the weights, model configuration, emotion mapping, and validation results was subsequently saved as:

```text
mgif_emotion_model.pt
```

The saved validation metadata is:

```python
{
    "macro_f1": 0.3737,
    "accuracy": 0.5940,
    "weighted_f1": 0.5749
}
```

---

## 7. Findings

An important result from this experiment is that the more sophisticated fusion mechanism did **not produce a substantial improvement over the simpler concatenation baseline**.

The validation results were approximately:

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Concatenation Baseline | **0.6251** | **0.3762** | **0.5917** |
| MGIF — Frozen Transformers | 0.5973 | 0.3712 | 0.5772 |
| MGIF — Fine-Tuned Transformers | 0.5940 | 0.3737 | 0.5749 |

Fine-tuning improved the MGIF model's Macro F1 from:

$$
0.3712 \rightarrow 0.3737
$$

However, the final MGIF result remained slightly below the concatenation baseline:

$$
0.3737 \text{ vs. } 0.3762
$$

The difference in Macro F1 is only:

$$
0.3762 - 0.3737 = 0.0025
$$

Therefore, this experiment does **not provide evidence that the additional fusion complexity improves validation performance for the current configuration**.

This is still an informative result.

The experiment demonstrates that introducing a more sophisticated fusion mechanism does not automatically produce better emotion recognition. Performance also depends on factors such as:

- the quality of the modality representations,
- dataset size and class imbalance,
- conversational context,
- regularization,
- optimization strategy,
- and how complementary the modalities are.

It is also notable that the simpler concatenation model achieved higher Accuracy and Weighted F1.

For the current embeddings and training configuration, directly preserving the contextual text and audio representations through concatenation appears to be at least as effective as the MGIF-inspired learned fusion mechanism.

Rather than discarding the MGIF experiment, it is retained as an architectural comparison and as evidence of the model-development process.

---

## 8. Current Model

The resulting MGIF architecture can be summarized as:

```text
             ┌─────────────────────────┐
Text ───────►│ Text Intra-Modal        │
Embeddings   │ Transformer             │
             └────────────┬────────────┘
                          │
                       h_text
                          │
                          ▼
                    ┌───────────┐
                    │           │
                    │   MGIF    │────► Fused Representation
                    │           │              │
                    └─────▲─────┘              ▼
                          │              Emotion Classifier
                       h_audio                  │
                          │                     ▼
             ┌────────────┴────────────┐    7 Emotions
Audio ──────►│ Audio Intra-Modal       │
Embeddings   │ Transformer             │
             └─────────────────────────┘
```

This model represents the multimodal **perception component** of the larger conversational robot pipeline.

---

## 9. Next Step: End-to-End Inference

Until this stage, the emotion model has been trained using precomputed BERT and Wav2Vec2 embeddings.

The next stage is to create an end-to-end inference pipeline capable of processing raw text and audio:

```text
Raw Text
   │
   ▼
 BERT
   │
   └─────────────────┐
                     │
                     ▼
              Rolling Context
                     │
                     ▼
          Intra-Modal Transformers
                     │
                     ▼
                    MGIF
                     │
                     ▼
               Emotion State
                     │
                     ▼
              Response Model
                     │
                     ▼
              Robot Response
                     ▲
                     │
Raw Audio            │
   │                 │
   ▼                 │
Wav2Vec2 ────────────┘
```

The emotion-recognition component will therefore act as the **perception layer** of the conversational system.

The predicted emotion can then be combined with the user's current utterance and conversational context and provided to a small language model responsible for generating an appropriate response.

---

## Summary

During this stage of the project:

1. Previously trained text and audio intra-modal Transformer weights were loaded.
2. A two-modality MGIF-inspired fusion mechanism was implemented.
3. Vector-grained and neuron-grained fusion were combined.
4. A seven-class MELD emotion classifier was added.
5. MGIF and the classifier were first trained while the Transformers were frozen.
6. The complete architecture was subsequently fine-tuned using smaller learning rates for the pre-trained Transformer components.
7. Early stopping and gradient clipping were used during fine-tuning.
8. The final MGIF model achieved a validation Macro F1 of **0.3737**.
9. The simpler concatenation baseline remained slightly better at **0.3762 Macro F1**.
10. The trained model was packaged into `mgif_emotion_model.pt` for local inference and integration into the final conversational pipeline.

Although MGIF did not significantly improve validation performance, the experiment provides a useful comparison between **simple feature concatenation and learned multimodal fusion**, and informs the architectural decisions used in the next stage of the project.
