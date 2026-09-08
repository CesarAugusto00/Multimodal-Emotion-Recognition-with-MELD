from collections import deque

import torch

from encoders.text_encoder import TextEncoder
from encoders.audio_encoder import AudioEncoder
from models.emotion_model import MGIFEmotionClassifier


class EmotionPipeline:

    def __init__(
        self,
        checkpoint_path,
        context_size=3,
        device=None
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.context_size = context_size

        # Raw-input encoders
        self.text_encoder = TextEncoder(device=self.device)
        self.audio_encoder = AudioEncoder(device=self.device)

        # Rolling conversational context
        self.text_buffer = deque(maxlen=context_size)
        self.audio_buffer = deque(maxlen=context_size)

        # Load trained checkpoint
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device
        )

        config = checkpoint["config"]

        self.model = MGIFEmotionClassifier(
            text_dim=config["text_dim"],
            audio_dim=config["audio_dim"],
            d_model=config["d_model"],
            num_classes=config["num_classes"],
            context_size=config["context_size"]
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.emotion_to_id = checkpoint["emotion_to_id"]

        self.id_to_emotion = {
            value: key
            for key, value in self.emotion_to_id.items()
        }


    def reset_context(self):
        """
        Reset conversation memory.
        """
        self.text_buffer.clear()
        self.audio_buffer.clear()


    def _prepare_context(self):
        """
        Convert rolling buffers into model tensors.

        Returns:
            text: [1, context_size, text_dim]
            audio: [1, context_size, audio_dim]
            mask: [1, context_size]
        """

        text_list = list(self.text_buffer)
        audio_list = list(self.audio_buffer)

        missing = self.context_size - len(text_list)

        text_dim = text_list[0].shape[-1]
        audio_dim = audio_list[0].shape[-1]

        text_padding = [
            torch.zeros(text_dim)
            for _ in range(missing)
        ]

        audio_padding = [
            torch.zeros(audio_dim)
            for _ in range(missing)
        ]

        text = torch.stack(
            text_padding + text_list
        )

        audio = torch.stack(
            audio_padding + audio_list
        )

        padding_mask = torch.tensor(
            [True] * missing +
            [False] * len(text_list),
            dtype=torch.bool
        )

        return (
            text.unsqueeze(0).to(self.device),
            audio.unsqueeze(0).to(self.device),
            padding_mask.unsqueeze(0).to(self.device)
        )


    def predict(self, text, audio_path):
        """
        Process one conversational utterance.

        Args:
            text: transcript string
            audio_path: path to corresponding audio file

        Returns:
            dictionary with emotion and confidence
        """

        # Step 1: Generate embeddings
        text_embedding = self.text_encoder.encode(text)
        audio_embedding = self.audio_encoder.encode(audio_path)

        # Step 2: Add to rolling context
        self.text_buffer.append(text_embedding)
        self.audio_buffer.append(audio_embedding)

        # Step 3: Prepare transformer input
        text_tensor, audio_tensor, mask = (
            self._prepare_context()
        )

        # Step 4: Emotion inference
        with torch.no_grad():

            logits = self.model(
                text_tensor,
                audio_tensor,
                mask
            )

            probabilities = torch.softmax(
                logits,
                dim=-1
            )

        prediction = probabilities.argmax(
            dim=-1
        ).item()

        confidence = probabilities.max().item()

        emotion = self.id_to_emotion[prediction]

        return {
            "emotion": emotion,
            "confidence": confidence,
            "text": text
        }