# Intra-Modal Transformation

## Overview

This stage develops and evaluates the **intra-modal component** of a multimodal emotion-recognition system. The purpose of this stage is to learn conversational context independently inside the **text** and **audio** modalities before introducing a more advanced cross-modal fusion method such as **Multi-Grained Interactive Fusion (MGIF)**.

The system uses previously extracted:

- **BERT text embeddings**
- **Wav2Vec2 audio embeddings**
- Dialogue and utterance metadata
- Emotion labels

Because the embeddings were already computed, BERT and Wav2Vec2 are not trained again during this stage. Instead, small Transformer encoders are trained on top of the stored utterance-level embeddings.

---

## Motivation

Emotion in conversation is not always determined by a single utterance. The interpretation of the current utterance may depend on what was said immediately before it.

For example, the same sentence can have a different emotional meaning depending on the preceding conversational context.

For this reason, the model does not process each utterance independently. A short rolling conversational context is maintained for each dialogue.

The goal of the intra-modal stage is therefore:

\[
\text{Text context} \rightarrow \text{Text Transformer}
\]

\[
\text{Audio context} \rightarrow \text{Audio Transformer}
\]

followed by a simple baseline fusion and emotion classifier.

This baseline provides a reference point that can later be compared with a more sophisticated MGIF-based fusion architecture.

---

## Conversational Context Buffer

A rolling context window of **3 utterances** is used.

For a dialogue containing:

```text
U1, U2, U3, U4, U5
```

the model receives the following windows:

```text
U1 -> [PAD, PAD, U1]
U2 -> [PAD, U1,  U2]
U3 -> [U1,  U2,  U3]
U4 -> [U2,  U3,  U4]
U5 -> [U3,  U4,  U5]
```

The most recent utterance is always the last element of the window.

A Python `deque` with `maxlen=3` is used to construct the rolling context:

```python
from collections import deque

context_size = 3

text_buffer = deque(maxlen=context_size)
audio_buffer = deque(maxlen=context_size)

samples = []
previous_dialogue = None

for i in range(len(df)):
    row = df.iloc[i]
    dialogue_id = row["Dialogue_ID"]

    # Reset context when a new dialogue starts
    if dialogue_id != previous_dialogue:
        text_buffer.clear()
        audio_buffer.clear()
        previous_dialogue = dialogue_id

    text_embedding = text_embeddings[i]
    audio_embedding = audio_embeddings[i]

    text_buffer.append(text_embedding)
    audio_buffer.append(audio_embedding)

    samples.append({
        "text": list(text_buffer),
        "audio": list(audio_buffer),
        "label": row["Emotion"],
        "dialogue": dialogue_id,
        "utterance": row["Utterance_ID"]
    })
```

The dataframe was first verified to be correctly ordered by `Dialogue_ID` and `Utterance_ID`.

The buffers are cleared whenever the dialogue changes. This prevents the context of one conversation from leaking into another.

---

## Training Sample Representation

For a full context window:

\[
[U_{t-2}, U_{t-1}, U_t]
\]

the model receives:

\[
[T_{t-2}, T_{t-1}, T_t]
\]

for text and:

\[
[A_{t-2}, A_{t-1}, A_t]
\]

for audio.

The target is the emotion of the current utterance:

\[
y_t = Emotion(U_t)
\]

Therefore, a training example can be represented as:

\[
([T_{t-2}, T_{t-1}, T_t],
[A_{t-2}, A_{t-1}, A_t])
\rightarrow y_t
\]

Ground-truth emotions from previous utterances are stored as metadata but are **not used as model inputs**.

---

## Padding

The first utterances in a dialogue do not yet have three available context elements.

Left padding is therefore used:

```text
[PAD, PAD, U1]
[PAD, U1,  U2]
[U1,  U2,  U3]
```

A padding mask informs the Transformer which positions should be ignored.

Because the current utterance is always placed at the final position, the contextualized representation for the current utterance can later be obtained with:

```python
current_representation = transformer_output[:, -1, :]
```

---

# Intra-Modal Transformer Architecture

Two independent Transformer encoders are used:

1. **Text Intra-Modal Transformer**
2. **Audio Intra-Modal Transformer**

The Transformers have the same internal architecture but separate learned parameters.

The text and audio streams are therefore contextualized independently before fusion.

## Transformer Configuration

The experimental configuration used in this baseline was:

| Parameter | Value |
|---|---:|
| Context size | 3 utterances |
| Transformer hidden dimension (`d_model`) | 256 |
| Attention heads | 4 |
| Transformer encoder layers | 1 |
| Feed-forward dimension | 512 |
| Dropout | 0.1 |
| Number of emotion classes | 7 |

Each modality first uses a linear projection to transform its original embedding dimension into the common Transformer dimension:

\[
Embedding_{input} \rightarrow 256
\]

Positional embeddings are then added so that the Transformer can distinguish the relative order of utterances inside the context window.

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn

class IntraModalTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=256,
        nhead=4,
        num_layers=1,
        dim_feedforward=512,
        dropout=0.1,
        context_size=3
    ):
        super().__init__()

        self.projection = nn.Linear(input_dim, d_model)

        self.position_embedding = nn.Embedding(
            context_size,
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x, padding_mask=None):
        x = self.projection(x)

        _, seq_len, _ = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        ).unsqueeze(0)

        x = x + self.position_embedding(positions)

        x = self.transformer(
            x,
            src_key_padding_mask=padding_mask
        )

        return x
```

---

# Baseline Multimodal Classifier

Before implementing MGIF, a simpler fusion mechanism was used as a **baseline**.

After the text and audio Transformers contextualize their respective sequences, only the representation corresponding to the current utterance is selected:

```python
current_text = text_context[:, -1, :]
current_audio = audio_context[:, -1, :]
```

Both representations have dimension:

\[
256
\]

They are concatenated:

\[
h_{combined} =
[h_{text};h_{audio}]
\]

which produces:

\[
256 + 256 = 512
\]

The concatenated representation is passed through a small feed-forward classifier:

```python
self.classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 7)
)
```

The complete baseline architecture is therefore:

```text
BERT embeddings
      |
      v
Text Intra-Modal Transformer
      |
      v
Current contextual text representation (256)
      |
      |---------------------\
                            |
                            v
                       Concatenation
                            |
                            v
                     [Text ; Audio]
                         (512)
                            |
                            v
                     MLP Classifier
                            |
                            v
                       7 Emotions
                            ^
                            |
      |---------------------/
      |
Current contextual audio representation (256)
      ^
      |
Audio Intra-Modal Transformer
      ^
      |
Wav2Vec2 embeddings
```

This classifier is intentionally simple. Its purpose is to establish a measurable baseline before replacing concatenation with **Multi-Grained Interactive Fusion**.

---

# Training

The model is trained using multiclass cross-entropy loss:

```python
criterion = nn.CrossEntropyLoss()
```

AdamW is used as the optimizer:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)
```

During this stage, the following components are trainable:

- Text embedding projection
- Text positional embeddings
- Text intra-modal Transformer
- Audio embedding projection
- Audio positional embeddings
- Audio intra-modal Transformer
- Baseline classification head

The pretrained BERT and Wav2Vec2 models are not updated because their embeddings were previously generated.

---

# Validation Strategy

The model is evaluated using:

- Validation Accuracy
- Macro F1
- Weighted F1

**Macro F1** is used as the primary checkpoint-selection metric because the emotion classes are not equally represented. Macro F1 gives equal importance to each emotion class and therefore provides a more informative view of performance on minority emotions.

Whenever validation Macro F1 improves, a new best checkpoint is saved.

---

# Experimental Results

Training was performed for 40 epochs.

The highest validation **Macro F1** was obtained at **epoch 21**.

| Metric | Best Result |
|---|---:|
| Best Epoch by Macro F1 | **21** |
| Validation Accuracy at Epoch 21 | **0.6045** |
| Macro F1 | **0.3762** |
| Weighted F1 | **0.5847** |

Other useful observations from the run:

| Metric | Best Value | Epoch |
|---|---:|---:|
| Validation Accuracy | **0.6251** | 17 |
| Macro F1 | **0.3762** | 21 |
| Weighted F1 | **0.5917** | 18 |

The model does not maximize all metrics at the same epoch. Because Macro F1 is the main model-selection criterion, the epoch-21 checkpoint is retained.

---

## Training Behavior

The training loss continuously decreased throughout training:

```text
Epoch 01 Loss: 1.4266
Epoch 10 Loss: 1.0759
Epoch 20 Loss: 0.9482
Epoch 30 Loss: 0.7934
Epoch 40 Loss: 0.5784
```

Validation performance initially improved, reaching its best Macro F1 at epoch 21:

```text
Epoch 21
Validation Accuracy: 0.6045
Macro F1:            0.3762
Weighted F1:         0.5847
```

After approximately epochs **21–24**, the training loss continued decreasing while validation performance stopped improving and generally deteriorated.

For example:

```text
Epoch 21
Loss:        0.9489
Macro F1:    0.3762

Epoch 40
Loss:        0.5784
Macro F1:    0.3418
```

This divergence indicates increasing **overfitting** during later training.

For this reason, the final model is not selected using the last training epoch. Instead, the best validation checkpoint is preserved.

---

# Saving the Intra-Modal Transformers

The baseline classification head is temporary.

The main reusable output from this experiment is the pair of trained intra-modal Transformer encoders.

Before saving them, the best checkpoint should be restored:

```python
model.load_state_dict(
    torch.load(
        "best_model.pt",
        map_location=device
    )
)
```

The text and audio Transformers can then be saved independently:

```python
torch.save(
    model.text_transformer.state_dict(),
    "text_intra_transformer.pt"
)

torch.save(
    model.audio_transformer.state_dict(),
    "audio_intra_transformer.pt"
)
```

A combined checkpoint can also store the model configuration and experiment metadata:

```python
checkpoint = {
    "text_transformer_state_dict":
        model.text_transformer.state_dict(),

    "audio_transformer_state_dict":
        model.audio_transformer.state_dict(),

    "config": {
        "text_dim": text_dim,
        "audio_dim": audio_dim,
        "d_model": 256,
        "nhead": 4,
        "num_layers": 1,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "context_size": 3
    },

    "validation_results": {
        "best_epoch": 21,
        "macro_f1": 0.3762,
        "accuracy": 0.6045,
        "weighted_f1": 0.5847
    }
}

torch.save(
    checkpoint,
    "intra_modal_transformers.pt"
)
```

---

# Role of This Model in the Final Architecture

This experiment should be interpreted as an **intra-modal baseline**, not the final multimodal model.

Current architecture:

```text
Text embeddings  -> Text Intra-Modal Transformer  --\
                                                    \
                                                     -> Concatenation -> Classifier
                                                    /
Audio embeddings -> Audio Intra-Modal Transformer --/
```

The next stage will replace the simple concatenation operation with **Multi-Grained Interactive Fusion (MGIF)**:

```text
Text embeddings  -> Text Intra-Modal Transformer  --\
                                                    \
                                                     -> MGIF -> Classifier
                                                    /
Audio embeddings -> Audio Intra-Modal Transformer --/
```

The saved intra-modal Transformer weights provide a trained initialization for the next stage.

Initially, the Transformer weights can be frozen while the fusion module and classifier learn to combine the already contextualized representations. They can then be unfrozen and fine-tuned jointly with MGIF using a smaller learning rate.

This allows the project to compare:

\[
\text{Intra-Modal Transformers + Concatenation}
\]

against:

\[
\text{Intra-Modal Transformers + MGIF}
\]

while keeping the context representation and evaluation protocol consistent.

---

# Conclusions

The intra-modal stage confirms that short conversational context can be modeled independently in the text and audio modalities using lightweight Transformer encoders.

A three-utterance rolling buffer was used to preserve recent conversational information while preventing context from crossing dialogue boundaries.

The baseline concatenation classifier achieved a best validation Macro F1 of **0.3762** at epoch 21. Later epochs showed signs of overfitting, so the best validation checkpoint rather than the final epoch is retained.

The trained text and audio Transformer weights are saved and will be reused as the initialization for the next stage of the architecture, where **Multi-Grained Interactive Fusion** will replace the simple concatenation baseline.
