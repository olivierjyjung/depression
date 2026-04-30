"""
MDTD: Multimodal Depression Temporal Detector
Configuration settings
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    # Encoder settings
    text_encoder: str = "roberta-base"
    speech_encoder: str = "openai/whisper-large-v3"
    freeze_encoders: bool = True  # Freeze pretrained encoders initially

    # Projection settings
    text_embed_dim: int = 768      # RoBERTa hidden size
    speech_embed_dim: int = 1280   # Whisper large encoder dim
    joint_embed_dim: int = 256     # Projected joint space dimension

    # LLM Rubric Branch settings
    num_llm_scores: int = 5        # engagement, detail, valence, warmth, specificity
    llm_score_embed_dim: int = 64  # Embedding dim for LLM scores

    # Temporal modeling
    lstm_hidden_dim: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    bidirectional: bool = True

    # Classification head
    classifier_hidden_dim: int = 128
    classifier_dropout: float = 0.3

    # Feature dimensions
    @property
    def block_feature_dim(self) -> int:
        """Per-block feature dimension: t + s + disagreement + llm_emb"""
        return self.joint_embed_dim * 2 + 1 + self.llm_score_embed_dim

    @property
    def lstm_output_dim(self) -> int:
        """LSTM output dimension"""
        return self.lstm_hidden_dim * (2 if self.bidirectional else 1)


@dataclass
class DataConfig:
    """Data loading configuration"""

    # Paths
    daic_woz_dir: str = "/Users/user/Desktop/DAIC-WOZ"
    audio_dir: str = "/Users/user/Desktop/DAIC-WOZ"  # Audio files location

    # Splits
    train_split: str = "train"
    dev_split: str = "dev"
    test_split: str = "test"

    # Target questions (18 open-ended questions)
    target_questions: List[str] = field(default_factory=lambda: [
        "how are you doing today",
        "how have you been feeling lately",
        "do you consider yourself an introvert",
        "how would your best friend describe you",
        "what would you say are some of your best qualities",
        "who's someone that's been a positive influence",
        "how close are you to your family",
        "tell me about your kids",
        "when was the last time you felt really happy",
        "what are some things you like to do for fun",
        "what do you do to relax",
        "what are some things that usually put you in a good mood",
        "when was the last time you argued with someone",
        "how are you at controlling your temper",
        "what do you do when you're annoyed",
        "what are some things that make you really mad",
        "what's one of your most memorable experiences",
        "is there anything you regret",
        "what are you most proud of",
        "how easy is it for you to get a good night's sleep",
    ])

    # Audio settings
    sample_rate: int = 16000
    max_audio_length: int = 30  # seconds

    # Text settings
    max_text_length: int = 512

    # Batching
    max_blocks_per_session: int = 20  # Max question blocks per interview


@dataclass
class LLMConfig:
    """LLM Rubric configuration"""

    # API settings
    api_provider: str = "openai"  # "openai", "anthropic", "local"
    api_endpoint: str = "https://api.openai.com/v1/chat/completions"
    model_name: str = "gpt-4"
    temperature: float = 0.0  # Deterministic for reproducibility

    # Rubric dimensions (order matters - matches model input)
    rubric_dimensions: List[str] = field(default_factory=lambda: [
        "engagement",           # 대화 참여도
        "detail_richness",      # 구체적 정보량
        "emotional_valence",    # 정서적 톤 (1=부정, 5=긍정)
        "social_warmth",        # 관계 묘사의 따뜻함
        "episodic_specificity", # 구체적 에피소드 vs 과잉일반화
    ])

    # Caching (LLM 호출 비용 절감)
    cache_dir: str = "./llm_cache"
    use_cache: bool = True

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class TrainConfig:
    """Training configuration"""

    # Basic training
    batch_size: int = 4
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01

    # Learning rate schedule
    warmup_steps: int = 500
    lr_scheduler: str = "cosine"  # "cosine", "linear", "constant"

    # Loss weights
    lambda_cls: float = 1.0       # Classification loss weight
    lambda_contra: float = 0.5    # Contrastive loss weight

    # Contrastive learning
    temperature: float = 0.07     # InfoNCE temperature

    # Regularization
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0

    # Early stopping
    patience: int = 10
    min_delta: float = 0.001

    # Checkpointing
    save_dir: str = "./checkpoints"
    save_every: int = 5  # Save every N epochs

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Reproducibility
    seed: int = 42

    # Logging
    log_every: int = 10  # Log every N steps
    wandb_project: Optional[str] = "mdtd-depression"
    wandb_run_name: Optional[str] = None


@dataclass
class EvalConfig:
    """Evaluation configuration"""

    # Thresholds
    depression_threshold: float = 0.5
    high_disagreement_threshold: float = 0.4

    # Metrics
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"
    ])

    # Interpretation
    generate_report: bool = True
    report_dir: str = "./reports"


@dataclass
class Config:
    """Main configuration combining all sub-configs"""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    def __post_init__(self):
        """Validate configuration"""
        assert self.model.joint_embed_dim > 0
        assert self.train.batch_size > 0
        assert 0 < self.train.temperature < 1
        assert self.model.num_llm_scores == len(self.llm.rubric_dimensions)


# Default configuration instance
default_config = Config()


def get_config(**overrides) -> Config:
    """Get configuration with optional overrides"""
    config = Config()

    for key, value in overrides.items():
        if "." in key:
            # Nested config: e.g., "model.joint_embed_dim"
            parts = key.split(".")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(config, key, value)

    return config
