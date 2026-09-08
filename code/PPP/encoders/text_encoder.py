# encoders/text_encoder.py

import torch
from transformers import AutoTokenizer, AutoModel


class TextEncoder:
    def __init__(
        self,
        model_name="distilbert-base-uncased",
        device="cpu"
    ):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        ).to(device)

        self.model.eval()

    def encode(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            outputs = self.model(**inputs)

        # CLS token representation
        embedding = outputs.last_hidden_state[:, 0, :]

        return embedding.squeeze(0).cpu()