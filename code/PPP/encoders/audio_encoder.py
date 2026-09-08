# encoders/audio_encoder.py

import torch
import librosa
from transformers import AutoProcessor, AutoModel


class AudioEncoder:
    def __init__(
        self,
        model_name="facebook/wav2vec2-base",
        device="cpu"
    ):
        self.device = device

        self.processor = AutoProcessor.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        ).to(device)

        self.model.eval()

    def encode(self, audio_path):
        audio, sr = librosa.load(
            audio_path,
            sr=16000
        )

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        input_values = inputs.input_values.to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(
                input_values
            )

        # Mean pooling across time
        embedding = (
            outputs.last_hidden_state.mean(dim=1)
        )

        return embedding.squeeze(0).cpu()