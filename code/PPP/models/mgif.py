import torch
import torch.nn as nn


class MGIF(nn.Module):

    def __init__(self, d_model=256):
        super().__init__()

        self.vector_score = nn.Linear(d_model, 1)

        self.neuron_gate = nn.Linear(
            d_model * 2,
            d_model
        )

        self.output_projection = nn.Linear(
            d_model * 2,
            d_model
        )

    def forward(self, text, audio):

        # Vector-grained fusion
        text_score = self.vector_score(text)
        audio_score = self.vector_score(audio)

        scores = torch.cat(
            [text_score, audio_score],
            dim=1
        )

        weights = torch.softmax(scores, dim=1)

        text_weight = weights[:, 0:1]
        audio_weight = weights[:, 1:2]

        vector_fused = (
            text_weight * text +
            audio_weight * audio
        )

        # Neuron-grained fusion
        combined = torch.cat(
            [text, audio],
            dim=-1
        )

        gate = torch.sigmoid(
            self.neuron_gate(combined)
        )

        neuron_fused = (
            gate * text +
            (1 - gate) * audio
        )

        # Combine both granularities
        fused = torch.cat(
            [vector_fused, neuron_fused],
            dim=-1
        )

        return self.output_projection(fused)