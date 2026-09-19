"""Variational autoencoder that compresses 384-d sentence embeddings to 256-d.

Pipeline (TASK 2):
    text --(TextEncoder, FROZEN)--> 384-d base embedding
         --(VAEHead)--> mu (256) / logvar (256) --(reparameterize)--> z (256)
         --(VAEDecoder)--> reconstructed 384-d embedding

Only ``VAEHead`` and ``VAEDecoder`` are trained; the sentence-transformers base
model is frozen (no gradient, kept in eval mode).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BASE_EMBED_DIM = 384
HEAD_HIDDEN_DIM = 512
LATENT_DIM = 256
DECODER_HIDDEN_DIM = 256
KL_WEIGHT = 0.001

TextInput = Union[str, Sequence[str]]


class TextEncoder(nn.Module):
    """Frozen ``all-MiniLM-L6-v2`` sentence encoder: text -> 384-d embedding."""

    def __init__(
        self,
        model_name: str = BASE_MODEL_NAME,
        device: Union[str, torch.device, None] = None,
    ) -> None:
        super().__init__()
        # Imported lazily so the module stays importable without the heavy dep.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name, device=str(device) if device is not None else None
        )
        # sentence-transformers >= 6 renamed the accessor; support both.
        get_dim = getattr(self.model, "get_embedding_dimension", None) or getattr(
            self.model, "get_sentence_embedding_dimension", None
        )
        self.embedding_dim = int(get_dim() or BASE_EMBED_DIM) if get_dim else BASE_EMBED_DIM
        self._freeze()

    # ------------------------------------------------------------------ utils
    def _freeze(self) -> None:
        """Put the base model in eval mode and disable all its gradients."""
        # sentence-transformers >= 3 wraps a ``transformers`` model; depending on
        # the version the wrapper is (or is not) an ``nn.Module``, so handle both.
        candidates: List[nn.Module] = []
        if isinstance(self.model, nn.Module):
            candidates.append(self.model)
        for attr in ("auto_model", "transformer", "model"):
            inner = getattr(self.model, attr, None)
            if isinstance(inner, nn.Module):
                candidates.append(inner)

        for module in candidates:
            module.eval()
            for param in module.parameters(recurse=True):
                param.requires_grad_(False)

    @property
    def is_frozen(self) -> bool:
        """True when no base-encoder parameter requires a gradient."""
        return all(not p.requires_grad for p in self.parameters())

    # ---------------------------------------------------------------- forward
    @torch.no_grad()
    def encode(
        self, texts: TextInput, batch_size: int = 64
    ) -> torch.Tensor:
        """Encode one string or a sequence of strings -> ``(N, 384)`` tensor."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        if not items:
            return torch.zeros((0, self.embedding_dim), dtype=torch.float32)

        chunks: List[torch.Tensor] = []
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            embeddings = self.model.encode(
                batch,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            chunks.append(torch.as_tensor(embeddings, dtype=torch.float32))

        return torch.cat(chunks, dim=0)

    def forward(self, texts: TextInput, batch_size: int = 64) -> torch.Tensor:
        return self.encode(texts, batch_size=batch_size)


class VAEHead(nn.Module):
    """384-d base embedding -> (mu, logvar) of a 256-d Gaussian posterior."""

    def __init__(
        self,
        input_dim: int = BASE_EMBED_DIM,
        hidden_dim: int = HEAD_HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        logvar_clamp: float = 10.0,
        logvar_bias: float = -6.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.logvar_clamp = logvar_clamp

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Posterior-collapse mitigation: start sigma = exp(0.5 * -6) ~ 0.05 so the
        # decoder can learn to use z before the reparameterisation noise grows.
        # Measured on this dataset (2000 examples, kl_weight = 0.001):
        #   logvar bias  0  -> recon 0.00210 (worse than the 0.00224 constant
        #                      predictor; the decoder simply ignores z)
        #   logvar bias -6  -> recon 0.00081 after 10 epochs, mu std 0.44
        if logvar_bias is not None:
            with torch.no_grad():
                self.logvar_layer.bias.fill_(logvar_bias)
                self.logvar_layer.weight.mul_(0.01)

    def forward(self, base_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.fc(base_embedding))
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        if self.logvar_clamp is not None:
            # Keeps exp(logvar) finite during early training.
            logvar = torch.clamp(logvar, -self.logvar_clamp, self.logvar_clamp)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + eps * exp(0.5 * logvar), eps ~ N(0, I)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class VAEDecoder(nn.Module):
    """256-d latent vector -> reconstructed 384-d base embedding."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        hidden_dim: int = DECODER_HIDDEN_DIM,
        output_dim: int = BASE_EMBED_DIM,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(z)))


class VAE(nn.Module):
    """``TextEncoder`` + ``VAEHead`` + ``VAEDecoder`` as a single module."""

    def __init__(
        self,
        encoder: Union[TextEncoder, None] = None,
        base_model_name: str = BASE_MODEL_NAME,
        input_dim: int = BASE_EMBED_DIM,
        hidden_dim: int = HEAD_HIDDEN_DIM,
        latent_dim: int = LATENT_DIM,
        decoder_hidden_dim: int = DECODER_HIDDEN_DIM,
        device: Union[str, torch.device, None] = None,
        logvar_bias: float = -6.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else TextEncoder(base_model_name, device)
        self.head = VAEHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            logvar_bias=logvar_bias,
        )
        self.decoder = VAEDecoder(
            latent_dim=latent_dim, hidden_dim=decoder_hidden_dim, output_dim=input_dim
        )

    # ---------------------------------------------------------------- plumbing
    def embed(self, texts: TextInput, batch_size: int = 64) -> torch.Tensor:
        """Frozen base embeddings only (no gradient) -> ``(N, 384)``."""
        return self.encoder(texts, batch_size=batch_size)

    def encode_from_embedding(
        self, base_embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """384-d embedding -> ``(z, mu, logvar)``."""
        mu, logvar = self.head(base_embedding)
        z = self.head.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """256-d latent -> reconstructed 384-d embedding."""
        return self.decoder(z)

    # ------------------------------------------------------------------- API
    def encode(
        self, text: TextInput, batch_size: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``text -> (z, mu, logvar)`` with the reparameterization trick."""
        return self.encode_from_embedding(self.embed(text, batch_size=batch_size))

    def forward(
        self, text: TextInput, batch_size: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``text -> (z, mu, logvar, reconstructed_embedding)``."""
        base_embedding = self.embed(text, batch_size=batch_size)
        mu, logvar = self.head(base_embedding)
        z = self.head.reparameterize(mu, logvar)
        return z, mu, logvar, self.decoder(z)

    @torch.no_grad()
    def encode_texts(self, texts: TextInput, batch_size: int = 64) -> torch.Tensor:
        """Deterministic latent encoding using ``mu`` (for retrieval/plots)."""
        _, mu, _ = self.encode(texts, batch_size=batch_size)
        return mu

    # ------------------------------------------------------------------ utils
    def trainable_parameters(self):
        """Iterator over head + decoder parameters (base encoder excluded)."""
        yield from self.head.parameters()
        yield from self.decoder.parameters()

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)


def vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = KL_WEIGHT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE objective: ``MSE + kl_weight * KL``.

    Both terms are reduced to a **per-element mean** so that ``kl_weight`` is a
    directly comparable, genuinely small weight (as the spec intends: 0.001).

    Why this matters (measured on this dataset): with an unnormalised
    ``torch.sum`` the KL term is ~384x stronger than the per-element MSE, which
    collapses the posterior. Concretely that drove ``mu`` to +/-0.017 (std
    0.002), ``logvar`` to ~0 and made reconstruction (MSE 0.002305) no better
    than predicting the mean embedding (MSE 0.002293) - i.e. ``z`` became pure
    noise. Using a mean for both terms removes that accidental 384x scaling.

    Returns ``(total, reconstruction_loss, kl_loss)``.
    """
    reconstruction_loss = F.mse_loss(reconstructed, original, reduction="mean")

    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

    total = reconstruction_loss + kl_weight * kl_loss
    return total, reconstruction_loss, kl_loss
