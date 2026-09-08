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