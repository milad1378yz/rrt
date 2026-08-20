"""Response Parameter Network used by Rubric Response Theory."""

import dataclasses
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from core.rrt import RRTConfig


DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"


class ResponseParameterNetwork(nn.Module):
    """Predict criterion discrimination and difficulty from prompt and criterion text.

    The text encoder is frozen. Its normalized embeddings are cached in memory,
    while two small MLPs predict ``a`` and ``b`` from ``[prompt; criterion]``.
    """

    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        *,
        hidden: int = 1024,
        layers: int = 3,
        dropout: float = 0.3,
        max_tokens: int = 3048,
        embed_batch_size: int = 64,
        rrt_config: RRTConfig = RRTConfig(),
    ):
        super().__init__()
        self.embed_model = str(embed_model)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.dropout_probability = float(dropout)
        self.max_tokens = int(max_tokens)
        self.embed_batch_size = int(embed_batch_size)
        self.rrt_config = rrt_config

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.embed_model,
            padding_side="left",
            trust_remote_code=True,
        )
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.embedder = AutoModel.from_pretrained(
            self.embed_model,
            dtype=dtype,
            trust_remote_code=True,
        )
        for parameter in self.embedder.parameters():
            parameter.requires_grad_(False)
        self.embedder.eval()

        input_width = 2 * self.embedder.config.hidden_size
        self.a_network = self._make_network(input_width)
        self.b_network = self._make_network(input_width)
        self._embedding_cache: dict[str, torch.Tensor] = {}

    def _make_network(self, input_width):
        modules = []
        width = input_width
        for _ in range(self.layers):
            modules.extend(
                [
                    nn.Linear(width, self.hidden),
                    nn.Tanh(),
                    nn.Dropout(self.dropout_probability),
                ]
            )
            width = self.hidden
        modules.append(nn.Linear(width, 1))
        return nn.Sequential(*modules)

    @property
    def device(self):
        return next(self.a_network.parameters()).device

    def train(self, mode: bool = True):
        super().train(mode)
        self.embedder.eval()
        return self

    @torch.no_grad()
    def _embed_missing(self, texts):
        for start in range(0, len(texts), self.embed_batch_size):
            chunk = texts[start : start + self.embed_batch_size]
            tokens = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            hidden = self.embedder(**tokens).last_hidden_state
            mask = tokens["attention_mask"]
            last_index = mask.sum(dim=1) - 1
            if self.tokenizer.padding_side == "left":
                last_index = torch.full_like(last_index, hidden.shape[1] - 1)
            rows = hidden[torch.arange(len(chunk), device=self.device), last_index]
            rows = nn.functional.normalize(rows.float(), dim=-1).cpu()
            self._embedding_cache.update(zip(chunk, rows))

    def embed_texts(self, texts):
        """Return cached frozen embeddings for a text sequence."""

        texts = [str(text) for text in texts]
        missing = list(dict.fromkeys(text for text in texts if text not in self._embedding_cache))
        if missing:
            self._embed_missing(missing)
        return torch.stack([self._embedding_cache[text] for text in texts]).to(self.device)

    def parameters_for_text(self, criteria, prompts):
        """Predict ``(a, b)`` for aligned criterion and prompt strings."""

        if len(criteria) != len(prompts):
            raise ValueError("criteria and prompts must have equal lengths")
        criterion_embeddings = self.embed_texts(criteria)
        prompt_embeddings = self.embed_texts(prompts)
        features = torch.cat([prompt_embeddings, criterion_embeddings], dim=-1)
        a = nn.functional.softplus(self.a_network(features).squeeze(-1))
        a = a + self.rrt_config.a_min
        b = self.rrt_config.b_bound * torch.tanh(self.b_network(features).squeeze(-1))
        return a, b

    @torch.no_grad()
    def predict(self, criteria, prompt):
        """Evaluation-mode NumPy predictions for one rubric."""

        was_training = self.training
        self.eval()
        prompts = [prompt] * len(criteria)
        a, b = self.parameters_for_text(criteria, prompts)
        self.train(was_training)
        return a.cpu().numpy(), b.cpu().numpy()

    def trainable_parameters(self):
        return [
            *self.a_network.parameters(),
            *self.b_network.parameters(),
        ]

    def checkpoint_config(self):
        return {
            "embed_model": self.embed_model,
            "hidden": self.hidden,
            "layers": self.layers,
            "dropout": self.dropout_probability,
            "max_tokens": self.max_tokens,
            "embed_batch_size": self.embed_batch_size,
            "rrt_config": dataclasses.asdict(self.rrt_config),
        }

    def save(self, path, **metadata):
        """Save the small RPN heads; the frozen text encoder is referenced by name."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": self.checkpoint_config(),
                "rpn": {
                    key: value
                    for key, value in self.state_dict().items()
                    if not key.startswith("embedder.")
                },
                **metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path, *, device=None):
        """Restore an RPN checkpoint and return ``(model, metadata)``."""

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = dict(checkpoint["config"])
        config["rrt_config"] = RRTConfig(**config["rrt_config"])
        model = cls(**config).to(device)
        model.load_state_dict(checkpoint["rpn"], strict=False)
        model.eval()
        return model, checkpoint
