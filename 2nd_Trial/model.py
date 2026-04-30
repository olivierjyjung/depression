"""
MDTD: Multimodal Depression Temporal Detector
Model architecture implementation

Two main ideas:
1. Cross-Modal Disagreement via Contrastive Alignment
2. Hybrid Temporal Dynamics (Encoder + LLM Rubric)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, WhisperModel
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from config import ModelConfig


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BlockFeatures:
    """Features extracted from a single question block"""
    text_emb: torch.Tensor       # (B, D) text embedding in joint space
    speech_emb: torch.Tensor     # (B, D) speech embedding in joint space
    disagreement: torch.Tensor   # (B, 1) cross-modal disagreement score
    combined: torch.Tensor       # (B, F) concatenated block feature


@dataclass
class ModelOutput:
    """Model output containing predictions and intermediate features"""
    prob: torch.Tensor                    # (B, 1) depression probability
    text_embs: torch.Tensor               # (B, N, D) text embeddings per block
    speech_embs: torch.Tensor             # (B, N, D) speech embeddings per block
    disagreements: torch.Tensor           # (B, N) disagreement scores per block
    mean_disagreement: torch.Tensor       # (B, 1) average disagreement
    disagreement_slope: torch.Tensor      # (B, 1) disagreement trend over time
    engagement_slope: torch.Tensor        # (B, 1) engagement trend over time (LLM)
    temporal_hidden: torch.Tensor         # (B, H) LSTM final hidden state
    llm_score_slopes: Optional[Dict[str, torch.Tensor]] = None  # LLM rubric slopes


# ============================================================================
# Encoder Modules
# ============================================================================

class TextEncoder(nn.Module):
    """Text encoder using RoBERTa with projection to joint space"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(config.text_encoder)
        self.projection = nn.Sequential(
            nn.Linear(config.text_embed_dim, config.joint_embed_dim),
            nn.LayerNorm(config.joint_embed_dim),
            nn.ReLU(),
            nn.Linear(config.joint_embed_dim, config.joint_embed_dim),
        )

        if config.freeze_encoders:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) attention mask

        Returns:
            (B, D) text embedding in joint space
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        projected = self.projection(cls_emb)  # (B, joint_dim)
        return projected


class SpeechEncoder(nn.Module):
    """Speech encoder using Whisper with projection to joint space"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(config.speech_encoder).encoder
        self.projection = nn.Sequential(
            nn.Linear(config.speech_embed_dim, config.joint_embed_dim),
            nn.LayerNorm(config.joint_embed_dim),
            nn.ReLU(),
            nn.Linear(config.joint_embed_dim, config.joint_embed_dim),
        )

        if config.freeze_encoders:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, mel_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_features: (B, 80, T) log-mel spectrogram

        Returns:
            (B, D) speech embedding in joint space
        """
        outputs = self.encoder(mel_features)
        pooled = outputs.last_hidden_state.mean(dim=1)  # (B, 1280)
        projected = self.projection(pooled)  # (B, joint_dim)
        return projected


# ============================================================================
# Temporal Modules
# ============================================================================

class TemporalEncoder(nn.Module):
    """Temporal modeling using Bi-LSTM for sequence of blocks"""

    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.lstm_hidden_dim,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True,
        )

    def forward(
        self,
        block_sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            block_sequence: (B, N, F) sequence of block features
            lengths: (B,) actual sequence lengths for packing

        Returns:
            outputs: (B, N, H) LSTM outputs at each timestep
            hidden: (B, H) final hidden state
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                block_sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            outputs, (h_n, _) = self.lstm(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        else:
            outputs, (h_n, _) = self.lstm(block_sequence)

        # Concatenate final hidden states from both directions
        if self.lstm.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, 2H)
        else:
            hidden = h_n[-1]  # (B, H)

        return outputs, hidden


class LLMScoreProjection(nn.Module):
    """
    Project LLM rubric scores into embedding space.

    LLM scores (engagement, detail, valence, etc.) are 1-5 scale values.
    This module converts them into learned embeddings for the model.
    """

    def __init__(self, num_scores: int, embed_dim: int):
        super().__init__()
        self.num_scores = num_scores

        # Normalize and project LLM scores
        self.projection = nn.Sequential(
            nn.Linear(num_scores, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, llm_scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            llm_scores: (B, num_scores) LLM rubric scores (1-5 scale)

        Returns:
            (B, embed_dim) projected embedding
        """
        # Normalize scores from 1-5 to 0-1
        normalized = (llm_scores - 1) / 4.0
        return self.projection(normalized)


# ============================================================================
# Classification Head
# ============================================================================

class ClassificationHead(nn.Module):
    """Classification head for depression prediction"""

    def __init__(self, input_dim: int, config: ModelConfig):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, config.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, F) concatenated features

        Returns:
            (B, 1) depression probability
        """
        return torch.sigmoid(self.classifier(features))


# ============================================================================
# Main Model
# ============================================================================

class DepressionDetector(nn.Module):
    """
    MDTD: Multimodal Depression Temporal Detector

    Combines two key ideas:
    1. Cross-Modal Disagreement via Contrastive Alignment
       - Text and speech projected to joint space
       - Disagreement = 1 - cosine_similarity

    2. Hybrid Temporal Dynamics
       - Encoder branch: automatic feature extraction
       - LLM Rubric branch: interpretable scores (engagement, detail, etc.)
       - Both tracked over time with LSTM
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # ===== Encoder Branch =====
        self.text_encoder = TextEncoder(config)
        self.speech_encoder = SpeechEncoder(config)

        # ===== LLM Rubric Branch =====
        # LLM scores: engagement, detail_richness, emotional_valence,
        #             social_warmth, episodic_specificity
        self.num_llm_scores = config.num_llm_scores
        self.llm_projection = LLMScoreProjection(
            num_scores=config.num_llm_scores,
            embed_dim=config.llm_score_embed_dim,
        )

        # ===== Block Feature Dimension =====
        # [text_emb; speech_emb; disagreement; llm_emb]
        block_feat_dim = (
            config.joint_embed_dim * 2 +  # text + speech
            1 +                            # disagreement
            config.llm_score_embed_dim     # LLM scores embedding
        )

        # ===== Temporal Modeling =====
        self.temporal_encoder = TemporalEncoder(block_feat_dim, config)

        # ===== Classification Head =====
        # Input: LSTM hidden + slopes (disagree, engagement, detail, valence)
        num_slopes = 1 + 3  # disagreement_slope + 3 LLM score slopes
        classifier_input_dim = config.lstm_output_dim + num_slopes + 1  # +1 for mean_disagree
        self.classifier = ClassificationHead(classifier_input_dim, config)

    def encode_block(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        speech_mel: torch.Tensor,
        llm_scores: torch.Tensor,
    ) -> BlockFeatures:
        """
        Encode a single question block with both encoder and LLM features.

        Args:
            text_input_ids: (B, L) tokenized text
            text_attention_mask: (B, L) attention mask
            speech_mel: (B, 80, T) mel spectrogram
            llm_scores: (B, num_scores) LLM rubric scores

        Returns:
            BlockFeatures with embeddings and disagreement
        """
        # Encoder branch
        text_emb = self.text_encoder(text_input_ids, text_attention_mask)  # (B, D)
        speech_emb = self.speech_encoder(speech_mel)  # (B, D)

        # Cross-modal disagreement (Idea 1)
        text_norm = F.normalize(text_emb, p=2, dim=-1)
        speech_norm = F.normalize(speech_emb, p=2, dim=-1)
        cosine_sim = (text_norm * speech_norm).sum(dim=-1, keepdim=True)
        disagreement = 1 - cosine_sim  # (B, 1)

        # LLM rubric branch (Idea 2)
        llm_emb = self.llm_projection(llm_scores)  # (B, llm_embed_dim)

        # Combine all features
        combined = torch.cat([text_emb, speech_emb, disagreement, llm_emb], dim=-1)

        return BlockFeatures(
            text_emb=text_emb,
            speech_emb=speech_emb,
            disagreement=disagreement,
            combined=combined,
        )

    def compute_slopes(
        self,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute temporal slope (trend) for a sequence of values.

        Args:
            values: (B, N) values over time
            mask: (B, N) valid position mask

        Returns:
            slopes: (B, 1) trend slope
        """
        B, N = values.shape
        device = values.device

        # Time indices centered at 0
        t = torch.arange(N, device=device, dtype=torch.float32)
        t = t - t.mean()

        if mask is not None:
            t = t.unsqueeze(0).expand(B, -1) * mask
            values_masked = values * mask
            t_sq_sum = (t ** 2).sum(dim=1, keepdim=True).clamp(min=1e-8)
            slopes = (values_masked * t).sum(dim=1, keepdim=True) / t_sq_sum
        else:
            t_sq_sum = (t ** 2).sum()
            slopes = (values * t).sum(dim=1, keepdim=True) / t_sq_sum

        return slopes

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        speech_mels: torch.Tensor,
        llm_scores: torch.Tensor,
        block_mask: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """
        Forward pass for full interview session.

        Args:
            text_input_ids: (B, N, L) tokenized text for each block
            text_attention_mask: (B, N, L) attention masks
            speech_mels: (B, N, 80, T) mel spectrograms for each block
            llm_scores: (B, N, num_scores) LLM rubric scores for each block
            block_mask: (B, N) mask for valid blocks

        Returns:
            ModelOutput with predictions and interpretable features
        """
        B, N, L = text_input_ids.shape
        device = text_input_ids.device

        # Process each block
        text_embs = []
        speech_embs = []
        disagreements = []
        block_features = []

        for i in range(N):
            block = self.encode_block(
                text_input_ids[:, i],
                text_attention_mask[:, i],
                speech_mels[:, i],
                llm_scores[:, i],
            )
            text_embs.append(block.text_emb)
            speech_embs.append(block.speech_emb)
            disagreements.append(block.disagreement)
            block_features.append(block.combined)

        # Stack along block dimension
        text_embs = torch.stack(text_embs, dim=1)           # (B, N, D)
        speech_embs = torch.stack(speech_embs, dim=1)       # (B, N, D)
        disagreements = torch.cat(disagreements, dim=1)     # (B, N)
        block_seq = torch.stack(block_features, dim=1)      # (B, N, F)

        # Temporal modeling
        lengths = block_mask.sum(dim=1) if block_mask is not None else None
        _, temporal_hidden = self.temporal_encoder(block_seq, lengths)

        # Compute slopes for various features
        disagree_slope = self.compute_slopes(disagreements, block_mask)

        # LLM score slopes (engagement, detail, valence)
        engagement_scores = llm_scores[:, :, 0]  # First LLM score = engagement
        detail_scores = llm_scores[:, :, 1]      # Second = detail_richness
        valence_scores = llm_scores[:, :, 2]     # Third = emotional_valence

        engagement_slope = self.compute_slopes(engagement_scores, block_mask)
        detail_slope = self.compute_slopes(detail_scores, block_mask)
        valence_slope = self.compute_slopes(valence_scores, block_mask)

        # Mean disagreement
        if block_mask is not None:
            mean_disagree = (disagreements * block_mask).sum(dim=1, keepdim=True) / \
                           block_mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            mean_disagree = disagreements.mean(dim=1, keepdim=True)

        # Combine all features for classification
        final_features = torch.cat([
            temporal_hidden,
            mean_disagree,
            disagree_slope,
            engagement_slope,
            detail_slope,
            valence_slope,
        ], dim=-1)

        # Classification
        prob = self.classifier(final_features)

        return ModelOutput(
            prob=prob,
            text_embs=text_embs,
            speech_embs=speech_embs,
            disagreements=disagreements,
            mean_disagreement=mean_disagree,
            disagreement_slope=disagree_slope,
            engagement_slope=engagement_slope,
            temporal_hidden=temporal_hidden,
            llm_score_slopes={
                "engagement": engagement_slope,
                "detail": detail_slope,
                "valence": valence_slope,
            },
        )


# ============================================================================
# Loss Functions
# ============================================================================

class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining classification and contrastive objectives.

    1. Classification Loss: BCE for depression prediction
    2. Contrastive Loss: InfoNCE for text-speech joint space alignment
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_contra: float = 0.5,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_contra = lambda_contra
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def classification_loss(self, prob: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Binary cross-entropy with optional label smoothing"""
        if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy(prob, labels)

    def contrastive_loss(
        self,
        text_embs: torch.Tensor,
        speech_embs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss for text-speech alignment.
        Same utterance pairs should be close, different pairs should be far.
        """
        B, N, D = text_embs.shape

        # Flatten to (B*N, D)
        text_flat = text_embs.reshape(-1, D)
        speech_flat = speech_embs.reshape(-1, D)

        if mask is not None:
            valid_mask = mask.reshape(-1).bool()
            text_flat = text_flat[valid_mask]
            speech_flat = speech_flat[valid_mask]

        # Normalize
        text_norm = F.normalize(text_flat, p=2, dim=-1)
        speech_norm = F.normalize(speech_flat, p=2, dim=-1)

        # Similarity matrix
        logits = torch.matmul(text_norm, speech_norm.T) / self.temperature

        # Labels: diagonal = positive pairs
        labels = torch.arange(len(text_norm), device=text_norm.device)

        # Symmetric loss
        loss_t2s = F.cross_entropy(logits, labels)
        loss_s2t = F.cross_entropy(logits.T, labels)

        return (loss_t2s + loss_s2t) / 2

    def forward(
        self,
        outputs: ModelOutput,
        labels: torch.Tensor,
        block_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Returns:
            Dictionary with total loss and individual components
        """
        loss_cls = self.classification_loss(outputs.prob, labels)
        loss_contra = self.contrastive_loss(
            outputs.text_embs, outputs.speech_embs, block_mask
        )

        total_loss = self.lambda_cls * loss_cls + self.lambda_contra * loss_contra

        return {
            "total": total_loss,
            "classification": loss_cls,
            "contrastive": loss_contra,
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_model(config: ModelConfig) -> DepressionDetector:
    """Factory function to create model"""
    return DepressionDetector(config)


def create_loss(
    lambda_cls: float = 1.0,
    lambda_contra: float = 0.5,
    temperature: float = 0.07,
    label_smoothing: float = 0.0,
) -> MultiTaskLoss:
    """Factory function to create loss"""
    return MultiTaskLoss(lambda_cls, lambda_contra, temperature, label_smoothing)
