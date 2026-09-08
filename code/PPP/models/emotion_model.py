import torch.nn as nn

from models.intramodal import IntraModalTransformer
from models.mgif import MGIF


class MGIFEmotionClassifier(nn.Module):

    def __init__(
        self,
        text_dim,
        audio_dim,
        d_model=256,
        num_classes=7,
        context_size=3
    ):
        super().__init__()

        self.text_transformer = IntraModalTransformer(
            input_dim=text_dim,
            d_model=d_model,
            context_size=context_size
        )

        self.audio_transformer = IntraModalTransformer(
            input_dim=audio_dim,
            d_model=d_model,
            context_size=context_size
        )

        self.mgif = MGIF(
            d_model=d_model
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        text,
        audio,
        padding_mask
    ):

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

        return self.classifier(fused)