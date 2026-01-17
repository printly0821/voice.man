"""
Training Data Models
SPEC-NLP-KOBERT-001 TAG-003: KoBERT Fine-tuning Models

Models for training configuration, checkpoints, and metrics.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json


class OptimizerType(str, Enum):
    """Supported optimizer types"""

    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class SchedulerType(str, Enum):
    """Supported learning rate scheduler types"""

    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    CONSTANT = "constant"


class LossType(str, Enum):
    """Supported loss function types"""

    CROSS_ENTROPY = "cross_entropy"
    FOCAL = "focal"
    LABEL_SMOOTHING = "label_smoothing"


@dataclass
class TrainingConfig:
    """
    Training configuration for KoBERT fine-tuning.

    Attributes:
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        train_val_split: Train/validation split ratio
        early_stopping_patience: Patience for early stopping
        early_stopping_min_delta: Minimum change to qualify as improvement
        max_length: Maximum sequence length for tokenization
        optimizer: Optimizer type
        scheduler: Learning rate scheduler type
        loss_type: Loss function type
        weight_decay: Weight decay for regularization
        warmup_steps: Number of warmup steps for learning rate
        gradient_accumulation_steps: Steps for gradient accumulation
        max_grad_norm: Maximum gradient norm for clipping
        num_labels: Number of emotion labels (6 or 7)
        emotion_labels: List of emotion label names
        seed: Random seed for reproducibility
        device: Device type ("auto", "cuda", "cpu")
        checkpoint_dir: Directory for saving checkpoints
        log_interval: Steps between logging
        eval_interval: Epochs between validation
        save_best_only: Only save best model based on validation loss
    """

    # Learning parameters
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 10
    train_val_split: float = 0.8

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.001

    # Model parameters
    max_length: int = 128
    num_labels: int = 6
    emotion_labels: List[str] = field(
        default_factory=lambda: ["happy", "sad", "angry", "fear", "disgust", "neutral"]
    )

    # Optimizer and scheduler
    optimizer: OptimizerType = OptimizerType.ADAMW
    scheduler: SchedulerType = SchedulerType.LINEAR
    loss_type: LossType = LossType.CROSS_ENTROPY
    weight_decay: float = 0.01
    warmup_steps: int = 500

    # Training stability
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Reproducibility
    seed: int = 42

    # Device and paths
    device: str = "auto"
    checkpoint_dir: str = "models/finetuned"

    # Logging and saving
    log_interval: int = 10
    eval_interval: int = 1
    save_best_only: bool = True

    def __post_init__(self):
        """Validate training configuration"""
        if not 0 < self.learning_rate < 1:
            raise ValueError(f"Learning rate must be between 0 and 1, got {self.learning_rate}")

        if not 0 < self.train_val_split < 1:
            raise ValueError(f"Train/val split must be between 0 and 1, got {self.train_val_split}")

        if self.num_labels != len(self.emotion_labels):
            raise ValueError(
                f"num_labels ({self.num_labels}) must match length of emotion_labels ({len(self.emotion_labels)})"
            )

        if self.num_labels not in [6, 7]:
            raise ValueError(f"num_labels must be 6 or 7, got {self.num_labels}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save config to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()
        # Convert enums to strings
        config_dict["optimizer"] = self.optimizer.value
        config_dict["scheduler"] = self.scheduler.value
        config_dict["loss_type"] = self.loss_type.value

        with open(path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load config from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        # Convert strings back to enums
        if "optimizer" in config_dict and isinstance(config_dict["optimizer"], str):
            config_dict["optimizer"] = OptimizerType(config_dict["optimizer"])
        if "scheduler" in config_dict and isinstance(config_dict["scheduler"], str):
            config_dict["scheduler"] = SchedulerType(config_dict["scheduler"])
        if "loss_type" in config_dict and isinstance(config_dict["loss_type"], str):
            config_dict["loss_type"] = LossType(config_dict["loss_type"])

        return cls(**config_dict)


@dataclass
class TrainingMetrics:
    """
    Training metrics for a single epoch.

    Attributes:
        epoch: Current epoch number
        train_loss: Average training loss for the epoch
        train_accuracy: Training accuracy
        val_loss: Average validation loss for the epoch
        val_accuracy: Validation accuracy
        learning_rate: Current learning rate
        epoch_time: Time taken for the epoch in seconds
    """

    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    epoch_time: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return asdict(self)


@dataclass
class TrainingHistory:
    """
    Training history containing metrics for all epochs.

    Attributes:
        config: Training configuration used
        metrics: List of epoch metrics
        started_at: Training start timestamp
        completed_at: Training completion timestamp
        best_epoch: Epoch with best validation loss
        best_val_loss: Best validation loss achieved
    """

    config: TrainingConfig
    metrics: List[TrainingMetrics] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    best_epoch: int = 0
    best_val_loss: float = float("inf")

    def add_metrics(self, metrics: TrainingMetrics) -> None:
        """Add metrics for an epoch"""
        self.metrics.append(metrics)

        # Update best epoch
        if metrics.val_loss < self.best_val_loss:
            self.best_val_loss = metrics.val_loss
            self.best_epoch = metrics.epoch

    def to_dict(self) -> Dict[str, Any]:
        """Convert history to dictionary"""
        return {
            "config": self.config.to_dict(),
            "metrics": [m.to_dict() for m in self.metrics],
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
        }


@dataclass
class ModelMetadata:
    """
    Model checkpoint metadata.

    Attributes:
        version: Model version string
        base_model: Base KoBERT model name
        emotion_labels: List of emotion labels
        num_labels: Number of emotion labels
        training_config: Training configuration used
        training_history: Training history summary
        trained_at: Training timestamp
        model_path: Path to model weights
        metrics: Final training metrics
    """

    version: str
    base_model: str
    emotion_labels: List[str]
    num_labels: int
    training_config: TrainingConfig
    trained_at: datetime
    model_path: str
    metrics: Optional[TrainingMetrics] = None
    training_history: Optional[TrainingHistory] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "version": self.version,
            "base_model": self.base_model,
            "emotion_labels": self.emotion_labels,
            "num_labels": self.num_labels,
            "training_config": self.training_config.to_dict(),
            "trained_at": self.trained_at.isoformat(),
            "model_path": self.model_path,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "training_history": self.training_history.to_dict() if self.training_history else None,
        }

    def to_json(self, path: Union[str, Path]) -> None:
        """Save metadata to JSON file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ModelMetadata":
        """Load metadata from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Reconstruct objects
        config = TrainingConfig.from_dict(data["training_config"])

        return cls(
            version=data["version"],
            base_model=data["base_model"],
            emotion_labels=data["emotion_labels"],
            num_labels=data["num_labels"],
            training_config=config,
            trained_at=datetime.fromisoformat(data["trained_at"]),
            model_path=data["model_path"],
        )


@dataclass
class CheckpointState:
    """
    Checkpoint state for resuming training.

    Attributes:
        epoch: Current epoch
        model_state_dict: Model state dict
        optimizer_state_dict: Optimizer state dict
        scheduler_state_dict: Scheduler state dict
        best_val_loss: Best validation loss so far
        best_epoch: Best epoch so far
        training_history: Training history
    """

    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]]
    best_val_loss: float
    best_epoch: int
    training_history: TrainingHistory

    def save(self, path: Union[str, Path]) -> None:
        """Save checkpoint to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "training_history": self.training_history.to_dict(),
        }

        torch_save(path, checkpoint)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CheckpointState":
        """Load checkpoint from file"""
        import torch

        checkpoint = torch.load(path, map_location="cpu")

        # Reconstruct training history
        config = TrainingConfig.from_dict(checkpoint["training_history"]["config"])
        history = TrainingHistory(config=config)
        history.started_at = (
            datetime.fromisoformat(checkpoint["training_history"]["started_at"])
            if checkpoint["training_history"].get("started_at")
            else None
        )
        history.completed_at = (
            datetime.fromisoformat(checkpoint["training_history"]["completed_at"])
            if checkpoint["training_history"].get("completed_at")
            else None
        )
        history.best_epoch = checkpoint["training_history"]["best_epoch"]
        history.best_val_loss = checkpoint["training_history"]["best_val_loss"]

        return cls(
            epoch=checkpoint["epoch"],
            model_state_dict=checkpoint["model_state_dict"],
            optimizer_state_dict=checkpoint["optimizer_state_dict"],
            scheduler_state_dict=checkpoint.get("scheduler_state_dict"),
            best_val_loss=checkpoint["best_val_loss"],
            best_epoch=checkpoint["best_epoch"],
            training_history=history,
        )


def torch_save(path: Path, obj: Any) -> None:
    """Helper function to save with torch"""
    import torch

    torch.save(obj, path)
