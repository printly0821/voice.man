# KLUE-BERT Model Integration

## Overview

This integration adds KLUE-BERT model support to the forensic NLP pipeline while maintaining full backward compatibility with the existing KoBERT implementation.

## What Was Implemented

### 1. KLUE-BERT Model Wrapper (`src/voice_man/services/nlp/klue_bert_model.py`)

A complete KLUE-BERT implementation with:
- Support for `klue/bert-base` and `klue/roberta-base` variants
- Singleton pattern for efficient resource usage
- GPU/CPU auto-detection and fallback
- Same API as KoBERT for easy swapping

**Key Features:**
```python
from voice_man.services.nlp.klue_bert_model import KLUEBERTModel, KLUEModelVariant

# Create KLUE-BERT model
model = KLUEBERTModel(
    model_name="klue/bert-base",  # or "klue/roberta-base"
    device="auto",
    max_length=128
)

# Use like KoBERT
outputs = model.encode("안녕하세요")
embeddings = model.get_embeddings("텍스트")
```

### 2. Unified BERT Interface (`src/voice_man/services/nlp/unified_bert.py`)

Factory pattern for model selection with:
- Configuration-based model switching
- A/B testing framework
- Performance benchmarking utilities
- Model caching for efficiency

**Key Features:**
```python
from voice_man.services.nlp.unified_bert import get_bert_model, BERTModelType

# Get model by type
model = get_bert_model(model_type="klue_bert")  # or "kobert", "klue_roberta"

# Use environment variable for auto-selection
# export VOICE_MAN_BERT_MODEL=klue_bert
model = get_bert_model()  # Auto-selects from env or config

# A/B testing
from voice_man.services.nlp.unified_bert import ABTestFramework

results = ABTestFramework.run_ab_test(
    texts=test_texts,
    model_a_type=BERTModelType.KOBERT,
    model_b_type=BERTModelType.KLUE_BERT,
)
```

### 3. Enhanced Emotion Classifier (`src/voice_man/services/nlp/emotion_classifier.py`)

Updated classifier with:
- Model selection support (KoBERT, KLUE-BERT, KLUE-RoBERTa)
- Auto-selection from configuration/environment
- Hot-swapping models
- Full backward compatibility

**Key Features:**
```python
from voice_man.services.nlp.emotion_classifier import UnifiedBERTEmotionClassifier

# Auto-select model (from env or config)
classifier = UnifiedBERTEmotionClassifier(model_type="auto")

# Explicit model selection
classifier = UnifiedBERTEmotionClassifier(model_type="klue_bert")

# Hot-swap models
classifier.swap_model("klue_roberta")

# Classify emotions
result = classifier.classify("정말 기쁜 하루입니다!")
print(result.primary_emotion)  # e.g., "happiness"
print(result.confidence)  # e.g., 0.92
```

### 4. Benchmarking Utilities (`src/voice_man/services/nlp/benchmark.py`)

Comprehensive benchmarking framework with:
- Performance comparison (latency, throughput, memory)
- A/B testing support
- Result storage and analysis
- Human-readable report generation

**Key Features:**
```python
from voice_man.services.nlp.benchmark import (
    BERTBenchmark,
    run_benchmark_comparison,
    generate_benchmark_report,
)

# Benchmark single model
metrics = BERTBenchmark.benchmark_model(model)

# Compare models
comparison = BERTBenchmark.compare_models(model_a, model_b)

# Run full comparison
results = run_benchmark_comparison(
    model_types=["kobert", "klue_bert", "klue_roberta"],
    output_dir=Path("ref/benchmark_results"),
)

# Generate report
generate_benchmark_report(results, Path("ref/benchmark_results/REPORT.md"))
```

### 5. Configuration (`src/voice_man/config/bert_config.yaml`)

Centralized configuration with:
- Model selection settings
- Environment variable override support
- A/B testing configuration
- Benchmarking settings

**Usage:**

1. **Using config file:**
   ```yaml
   model:
     type: "klue_bert"  # Options: kobert, klue_bert, klue_roberta
     device: "auto"
     max_length: 128
   ```

2. **Using environment variable:**
   ```bash
   export VOICE_MAN_BERT_MODEL=klue_bert
   python your_app.py
   ```

3. **Programmatic access:**
   ```python
   from voice_man.config import get_bert_config

   config = get_bert_config()
   model_type = config.get_model_type()
   model_name = config.get_model_name()
   device = config.get_device()
   ```

## Model Comparison

| Feature | KoBERT | KLUE-BERT | KLUE-RoBERTa |
|---------|--------|-----------|--------------|
| Model Name | `skt/kobert-base-v1` | `klue/bert-base` | `klue/roberta-base` |
| Parameters | 110M | 110M | 125M |
| Architecture | BERT | BERT | RoBERTa |
| Hidden Size | 768 | 768 | 768 |
| Training Data | Korean corpus | Korean corpus (KLUE) | Korean corpus (KLUE) |
| Release Date | 2020 | 2021 | 2021 |

## Migration Guide

### For Existing KoBERT Code

**Before:**
```python
from voice_man.services.nlp.emotion_classifier import KoBERTEmotionClassifier

classifier = KoBERTEmotionClassifier(
    model_name="skt/kobert-base-v1",
    device="auto"
)
```

**After (no changes needed - backward compatible):**
```python
from voice_man.services.nlp.emotion_classifier import KoBERTEmotionClassifier

# Same code works - defaults to KoBERT
classifier = KoBERTEmotionClassifier()
```

**To use KLUE-BERT:**
```python
from voice_man.services.nlp.emotion_classifier import UnifiedBERTEmotionClassifier

classifier = UnifiedBERTEmotionClassifier(model_type="klue_bert")
```

### Using Environment Variable

```bash
# System-wide model selection
export VOICE_MAN_BERT_MODEL=klue_bert

# Or per-session
VOICE_MAN_BERT_MODEL=klue_roberta python app.py
```

## Running Benchmarks

```bash
# Run Python script to benchmark all models
python -c "
from voice_man.services.nlp.benchmark import run_benchmark_comparison, generate_benchmark_report
from pathlib import Path

results = run_benchmark_comparison(
    model_types=['kobert', 'klue_bert'],
    output_dir=Path('ref/benchmark_results')
)

generate_benchmark_report(results, Path('ref/benchmark_results/REPORT.md'))
"
```

## File Structure

```
src/voice_man/
├── services/
│   └── nlp/
│       ├── klue_bert_model.py       # KLUE-BERT wrapper
│       ├── kobert_model.py          # Existing KoBERT (unchanged)
│       ├── unified_bert.py          # Factory & A/B testing
│       ├── emotion_classifier.py    # Updated with model selection
│       └── benchmark.py             # Benchmarking utilities
├── config/
│   ├── bert_config.yaml             # BERT configuration
│   └── __init__.py                  # Config manager
ref/
└── benchmark_results/               # Benchmark output directory
```

## Testing

### Test Model Loading

```python
from voice_man.services.nlp.unified_bert import get_bert_model

# Test each model
for model_type in ["kobert", "klue_bert", "klue_roberta"]:
    model = get_bert_model(model_type=model_type)
    assert model.is_loaded()
    print(f"{model_type}: OK")
```

### Test Emotion Classification

```python
from voice_man.services.nlp.emotion_classifier import UnifiedBERTEmotionClassifier

for model_type in ["kobert", "klue_bert"]:
    classifier = UnifiedBERTEmotionClassifier(model_type=model_type)
    result = classifier.classify("정말 기분이 좋아요!")
    print(f"{model_type}: {result.primary_emotion} ({result.confidence:.2f})")
```

### Test Model Swapping

```python
from voice_man.services.nlp.emotion_classifier import UnifiedBERTEmotionClassifier

classifier = UnifiedBERTEmotionClassifier(model_type="kobert")
info_before = classifier.get_current_model_info()
print(f"Before: {info_before['model_type']}")

# Swap to KLUE-BERT
success = classifier.swap_model("klue_bert")
info_after = classifier.get_current_model_info()
print(f"After: {info_after['model_type']}")
```

## Performance Considerations

1. **First Run:** Model download happens on first use (~300-400MB per model)
2. **GPU Memory:** Each model uses ~1-2GB GPU memory
3. **Caching:** Models are cached in memory after first load
4. **Batch Processing:** Use `classify_batch()` for better throughput

## Troubleshooting

### Model Download Issues

If model download fails:
```bash
# Set HuggingFace cache directory
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

### Out of Memory

If GPU OOM occurs:
```python
# Use CPU
classifier = UnifiedBERTEmotionClassifier(model_type="klue_bert", device="cpu")

# Or reduce batch size
classifier = UnifiedBERTEmotionClassifier(model_type="klue_bert")
results = classifier.classify_batch(texts, batch_size=1)  # Reduce from 8
```

### Model Not Found

If model name is incorrect:
```python
from voice_man.services.nlp.unified_bert import BERTModelFactory

# Check available models
print(BERTModelFactory.DEFAULT_CONFIGS)
```

## Next Steps

1. **Run Benchmarks:** Compare KoBERT vs KLUE-BERT on your data
2. **Update Config:** Set default model in `bert_config.yaml`
3. **Fine-tune:** Train KLUE-BERT on forensic domain data
4. **Deploy:** Use environment variable for production selection

## Summary

This KLUE-BERT integration provides:
- Full backward compatibility with existing KoBERT code
- Easy model swapping via configuration or environment
- Comprehensive benchmarking and A/B testing
- Production-ready configuration management
- Extensive documentation and examples

All existing code continues to work without changes, while new code can leverage KLUE-BERT's potentially improved performance on Korean NLP tasks.
