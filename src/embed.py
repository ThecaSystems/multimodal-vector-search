import torch
from transformers import AutoTokenizer, AutoModel


class TextEmbedder:
    def __init__(self, model_name: str) -> None:
        """Initializes the text embedder with the specified language model."""

        self.model_name = model_name
        self.device = self.get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    @staticmethod
    def get_device() -> torch.device:
        """Get device to run the model on."""
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        return device

    def embed(self, text: str) -> torch.FloatTensor:
        """Generates a text embedding for the given text."""

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state
            pooled_embeddings = embeddings.mean(dim=1)

        return pooled_embeddings.cpu()
