# KoBERT Fine-Tuning System Documentation

## Overview

The KoBERT Fine-Tuning System provides comprehensive capabilities for training custom Korean emotion classification models on top of the pre-trained KoBERT base model. This system is designed for forensic voice analysis applications requiring accurate emotion detection.

**Features:**

- Automatic GPU/CPU detection and optimization
- Support for 6 or 7 emotion categories
- NSMC dataset integration with automatic adaptation
- Training loop with validation and early stopping
- Model checkpointing and version tracking
- Learning rate scheduling and gradient accumulation
- Comprehensive logging and metrics

## Installation

Ensure you have the required dependencies:

```bash
pip install torch transformers korpora tqdm numpy
```

## Quick Start

### 1. Basic Training with Default Configuration

```python
from voice_man.services.nlp.training import KoBERTFineTuner, create_default_config

# Create default configuration for 6 emotions
config = create_default_config(num_labels=6)

# Initialize fine-tuner
fine_tuner = KoBERTFineTuner(config)

# Train the model using NSMC dataset (automatically loaded)
history = fine_tuner.train()

# Save the fine-tuned model
model_path = fine_tuner.save_model("v1.0.0")
print(f"Model saved to: {model_path}")
```

### 2. Using a Fine-Tuned Model for Inference

```python
from voice_man.services.nlp.emotion_classifier import KoBERTEmotionClassifier

# Initialize with fine-tuned model
classifier = KoBERTEmotionClassifier(
    fine_tuned_path="models/finetuned/kobert_emotion_v1.0.0.pt",
    num_labels=6
)

# Classify emotion
result = classifier.classify("정말 기뻐요!")
print(f"Primary emotion: {result.primary_emotion}")
print(f"Confidence: {result.confidence:.2f}")
print(f"All scores: {result.emotion_scores}")
```

### 3. Custom Training Configuration

```python
from voice_man.services.nlp.training import KoBERTFineTuner
from voice_man.models.nlp.training import TrainingConfig, OptimizerType, SchedulerType

# Create custom configuration
config = TrainingConfig(
    num_labels=6,
    emotion_labels=["happy", "sad", "angry", "fear", "disgust", "neutral"],
    num_epochs=10,
    batch_size=32,
    learning_rate=3e-5,
    early_stopping_patience=5,
    optimizer=OptimizerType.ADAMW,
    scheduler=SchedulerType.COSINE,
)

# Train with custom configuration
fine_tuner = KoBERTFineTuner(config)
history = fine_tuner.train()
```

## Training Configuration

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 2e-5 | Learning rate for optimizer |
| `batch_size` | int | 16 | Batch size for training |
| `num_epochs` | int | 10 | Number of training epochs |
| `train_val_split` | float | 0.8 | Train/validation split ratio |
| `early_stopping_patience` | int | 3 | Patience for early stopping |
| `max_length` | int | 128 | Maximum sequence length |
| `num_labels` | int | 6 | Number of emotion labels |
| `emotion_labels` | List[str] | 6 emotions | Emotion label names |
| `optimizer` | OptimizerType | ADAMW | Optimizer type |
| `scheduler` | SchedulerType | LINEAR | LR scheduler type |
| `weight_decay` | float | 0.01 | Weight decay for regularization |
| `warmup_steps` | int | 500 | Number of warmup steps |
| `device` | str | "auto" | Device type ("auto", "cuda", "cpu") |

### Optimizer Types

- `ADAMW`: AdamW optimizer (recommended)
- `ADAM`: Adam optimizer
- `SGD`: SGD with momentum

### Scheduler Types

- `LINEAR`: Linear warmup with linear decay
- `COSINE`: Cosine annealing
- `EXPONENTIAL`: Exponential decay
- `CONSTANT`: Constant learning rate

## Dataset Integration

### Using NSMC Dataset

The NSMC (Naver Sentiment Movie Corpus) dataset is automatically loaded and adapted for emotion classification:

```python
# Train with NSMC (default: 10,000 samples)
history = fine_tuner.train(use_nsmc=True, nsmc_samples=10000)
```

### Using Custom Data

```python
# Your custom training data
train_texts = ["정말 기뻐요", "너무 슬퍼요", ...]
train_labels = [0, 1, ...]  # 0=happy, 1=sad, etc.

val_texts = [...]
val_labels = [...]

# Train with custom data
history = fine_tuner.train(
    train_texts=train_texts,
    train_labels=train_labels,
    val_texts=val_texts,
    val_labels=val_labels,
    use_nsmc=False
)
```

## Model Management

### Saving Models

Models are saved with complete metadata:

```python
# Save with version
model_path = fine_tuner.save_model("v1.0.0")

# This creates:
# - models/finetuned/kobert_emotion_v1.0.0.pt (model weights)
# - models/finetuned/kobert_emotion_v1.0.0_metadata.json (metadata)
# - models/finetuned/kobert_emotion_v1.0.0_history.json (training history)
```

### Loading Models

```python
from voice_man.services.nlp.training import KoBERTFineTuner

# Load fine-tuned model
classification_head, metadata = KoBERTFineTuner.load_finetuned_model(
    "models/finetuned/kobert_emotion_v1.0.0.pt"
)

print(f"Model version: {metadata.version}")
print(f"Emotion labels: {metadata.emotion_labels}")
print(f"Training accuracy: {metadata.metrics.val_accuracy:.2f}")
```

### Model Metadata

Metadata file includes:

```json
{
  "version": "1.0.0",
  "base_model": "skt/kobert-base-v1",
  "emotion_labels": ["happy", "sad", "angry", "fear", "disgust", "neutral"],
  "num_labels": 6,
  "training_config": {...},
  "trained_at": "2026-01-13T10:30:00",
  "model_path": "models/finetuned/kobert_emotion_v1.0.0.pt",
  "metrics": {
    "epoch": 5,
    "train_loss": 0.3245,
    "train_accuracy": 0.8765,
    "val_loss": 0.3891,
    "val_accuracy": 0.8543,
    "learning_rate": 1.5e-5,
    "epoch_time": 120.5
  }
}
```

## Inference with Fine-Tuned Models

### Basic Usage

```python
from voice_man.services.nlp.emotion_classifier import KoBERTEmotionClassifier

# Option 1: Load fine-tuned model during initialization
classifier = KoBERTEmotionClassifier(
    fine_tuned_path="models/finetuned/kobert_emotion_v1.0.0.pt",
    num_labels=6
)

# Option 2: Load pre-trained model and reload fine-tuned later
classifier = KoBERTEmotionClassifier(num_labels=6)
classifier.reload_finetuned_model("models/finetuned/kobert_emotion_v1.0.0.pt")

# Classify emotion
result = classifier.classify("정말 화가 나요!")
```

### Batch Classification

```python
texts = [
    "정말 기뻐요",
    "너무 슬퍼요",
    "화가 나요",
    "무서워요",
]

results = classifier.classify_batch(texts, batch_size=4)

for text, result in zip(texts, results):
    print(f"{text} -> {result.primary_emotion} ({result.confidence:.2f})")
```

### Fallback Behavior

If fine-tuned model loading fails, the classifier automatically falls back to the pre-trained KoBERT model with a warning:

```python
# This will use pre-trained model if fine_tuned_path doesn't exist
classifier = KoBERTEmotionClassifier(
    fine_tuned_path="nonexistent_path.pt",
    num_labels=6
)
# Warning logged: "Fine-tuned model not found, using pre-trained model"
```

## Emotion Categories

### 6-Emotion Model (Default)

- `happy`: Happiness/joy (행복)
- `sad`: Sadness/grief (슬픔)
- `angry`: Anger/rage (분노)
- `fear`: Fear/terror (공포)
- `disgust`: Disgust/revulsion (혐오)
- `neutral`: Neutral/calm (중립)

### 7-Emotion Model (SPEC-NLP-KOBERT-001)

- `happiness`: Happiness/joy (행복)
- `sadness`: Sadness/grief (슬픔)
- `anger`: Anger/rage (분노)
- `fear`: Fear/terror (공포)
- `disgust`: Disgust/revulsion (혐오)
- `surprise`: Surprise/shock (놀람)
- `neutral`: Neutral/calm (중립)

## Training Tips

### For Better Accuracy

1. **Use more training data**: Increase `nsmc_samples` or provide custom labeled data
2. **Adjust learning rate**: Lower LR (1e-5) for more stable training
3. **Increase epochs**: Train for more epochs with early stopping
4. **Use larger batch size**: If GPU memory allows

```python
config = TrainingConfig(
    learning_rate=1e-5,
    batch_size=32,
    num_epochs=15,
    early_stopping_patience=5,
)
```

### For Faster Training

1. **Use smaller batch size**: Reduces memory usage
2. **Reduce max_length**: Shorter sequences process faster
3. **Enable gradient accumulation**: Simulate larger batch size

```python
config = TrainingConfig(
    batch_size=8,
    max_length=64,
    gradient_accumulation_steps=4,
)
```

### GPU Memory Management

If you encounter CUDA out-of-memory errors:

```python
# Reduce batch size
config.batch_size = 8

# Reduce sequence length
config.max_length = 64

# Use gradient accumulation
config.gradient_accumulation_steps = 4  # Effective batch size = 8 * 4 = 32
```

## Training Output

### Console Logs

```
2026-01-13 10:30:00 - INFO - Loading KoBERT model: skt/kobert-base-v1
2026-01-13 10:30:05 - INFO - Model loaded on cuda
2026-01-13 10:30:05 - INFO - Loading NSMC dataset with 10000 samples
2026-01-13 10:30:10 - INFO - Loaded 10000 training samples from NSMC
2026-01-13 10:30:10 - INFO - Split data: 8000 train, 2000 validation samples
2026-01-13 10:30:10 - INFO - Starting training for 5 epochs
2026-01-13 10:30:10 - INFO - Device: cuda
2026-01-13 10:30:10 - INFO - Batch size: 16
Epoch 1/5: 100%|██████████| 500/500 [02:15<00:00, 3.68it/s, loss=0.5234, acc=0.8123]
2026-01-13 10:32:25 - INFO - Epoch 1/5 - train_loss: 0.5234, train_acc: 0.8123, val_loss: 0.4567, val_acc: 0.8456
...
```

### Training History

Access training metrics after training:

```python
history = fine_tuner.train()

# Get best epoch
print(f"Best epoch: {history.best_epoch}")
print(f"Best val loss: {history.best_val_loss:.4f}")

# Get all metrics
for metrics in history.metrics:
    print(f"Epoch {metrics.epoch}: val_acc={metrics.val_accuracy:.4f}")
```

## Troubleshooting

### Model Loading Issues

**Problem**: `FileNotFoundError` when loading fine-tuned model

**Solution**: Ensure the model path exists and metadata file is present:

```python
from pathlib import Path

model_path = Path("models/finetuned/kobert_emotion_v1.0.0.pt")
metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"

assert model_path.exists(), "Model file not found"
assert metadata_path.exists(), "Metadata file not found"
```

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size or use CPU:

```python
config = TrainingConfig(
    batch_size=8,  # Reduce from 16
    device="cpu",  # Force CPU
)
```

### NSMC Dataset Not Loading

**Problem**: `Korpora package not installed`

**Solution**: Install Korpora:

```bash
pip install Korpora
```

## Advanced Usage

### Custom Loss Function

```python
from voice_man.models.nlp.training import LossType

config = TrainingConfig(
    loss_type=LossType.LABEL_SMOOTHING,  # Label smoothing for better generalization
)
```

### Learning Rate Scheduling

```python
from voice_man.models.nlp.training import SchedulerType

config = TrainingConfig(
    scheduler=SchedulerType.COSINE,  # Cosine annealing
    warmup_steps=1000,  # More warmup
)
```

### Resume Training from Checkpoint

```python
# Load checkpoint and continue training
state = CheckpointState.load("models/finetuned/checkpoint.pt")

# Reinitialize fine-tuner with config from checkpoint
fine_tuner = KoBERTFineTuner(state.training_history.config)

# Resume training
history = fine_tuner.train()
```

## Production Deployment

### Model Versioning

Use semantic versioning for models:

```python
# Major.Minor.Patch
# - Major: Architecture changes (e.g., 6 -> 7 emotions)
# - Minor: Retraining with same config
# - Patch: Bug fixes

fine_tuner.save_model("v1.2.3")
```

### Model Registry

Track models in a registry:

```python
MODELS = {
    "v1.0.0": {
        "path": "models/finetuned/kobert_emotion_v1.0.0.pt",
        "num_labels": 6,
        "accuracy": 0.8543,
        "trained_at": "2026-01-13",
    },
    "v1.1.0": {
        "path": "models/finetuned/kobert_emotion_v1.1.0.pt",
        "num_labels": 6,
        "accuracy": 0.8721,
        "trained_at": "2026-01-15",
    },
}
```

### A/B Testing

Compare models:

```python
# Load two models
model_v1 = KoBERTEmotionClassifier(
    fine_tuned_path="models/finetuned/kobert_emotion_v1.0.0.pt",
    num_labels=6
)
model_v2 = KoBERTEmotionClassifier(
    fine_tuned_path="models/finetuned/kobert_emotion_v1.1.0.pt",
    num_labels=6
)

# Compare predictions
text = "정말 기뻐요!"
result_v1 = model_v1.classify(text)
result_v2 = model_v2.classify(text)

print(f"v1: {result_v1.primary_emotion} ({result_v1.confidence:.2f})")
print(f"v2: {result_v2.primary_emotion} ({result_v2.confidence:.2f})")
```

## References

- **SPEC-NLP-KOBERT-001**: Complete specification for KoBERT integration
- **KoBERT Paper**: [Learning Korean Representations with BERT](https://arxiv.org/abs/2004.07077)
- **NSMC Dataset**: [Naver Sentiment Movie Corpus](https://github.com/e9t/nsmc)
- **Korpora**: [Korean Dataset Collection](https://github.com/ko-nlp/Korpora)

## Support

For issues or questions:
- Check the documentation in `docs/KOBERT_FINETUNING.md`
- Review example usage in `examples/finetuning_example.py`
- Consult SPEC-NLP-KOBERT-001 for detailed requirements
