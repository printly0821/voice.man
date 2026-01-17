# Memory-Safe Batch Processing Documentation

## Overview

The Memory-Safe Batch Processing system provides comprehensive fault-tolerant batch processing with advanced memory management and checkpoint recovery. This system integrates the new `MemoryManager` and enhanced `CheckpointManager` to provide reliable, resumable batch processing for voice analysis workflows.

## Key Features

### 1. Advanced Memory Management

#### New MemoryManager Integration
- **Pre-allocation Memory Checks**: Verifies sufficient memory before starting each batch
- **Per-File Memory Tracking**: Tracks memory usage for individual files with detailed statistics
- **Mid-Processing Monitoring**: Background watchdog monitors memory pressure during processing
- **Complete Service Cleanup**: Comprehensive cleanup of all registered services between batches
- **Memory Pressure Prediction**: Predicts OOM conditions using historical data

#### Memory Thresholds
- **System Memory Threshold**: Default 85% (configurable via `--memory-threshold`)
- **GPU Memory Threshold**: Default 90% (configurable via `--gpu-threshold`)
- **Critical Threshold**: 95% (triggers emergency cleanup)
- **Safety Margin**: 30% buffer for pre-allocation checks

### 2. Enhanced Checkpoint System

#### Checkpoint Validation
- **Pre-Resume Validation**: Validates checkpoint integrity before resuming
- **Database Integrity Checks**: Verifies database schema and data consistency
- **File Existence Validation**: Checks that all tracked files still exist
- **Automatic Repair**: Repairs minor checkpoint corruptions automatically

#### Resume Capabilities
- **Automatic Workflow Detection**: Finds and loads the latest crashed/running workflow
- **Smart File Filtering**: Skips already processed and failed files
- **Batch Continuation**: Continues from the last completed batch
- **State Preservation**: Preserves all processing state for seamless resume

### 3. Error Handling and Recovery

#### Error Classification
- **Transient Errors**: Temporary issues (network, temporary GPU unavailability)
- **Permanent Errors**: Fatal issues (corrupted files, invalid format)
- **Resource Errors**: Memory/disk space issues
- **Service Errors**: Service-specific failures

#### Intelligent Retry Strategy
- **Exponential Backoff**: Increasing delays between retries
- **Cleanup Before Retry**: Memory cleanup before retry attempts
- **CPU Fallback**: Falls back to CPU if GPU fails
- **Max Retry Limit**: Configurable retry attempts (default: 3)

#### OOM Recovery
- **Batch Size Reduction**: Reduces batch size by 50% on OOM
- **Maximum OOM Retries**: Up to 3 OOM recovery attempts
- **Emergency Cleanup**: Aggressive cleanup when OOM is detected
- **Minimum Batch Size**: Stops if batch size reaches 1 (minimum)

### 4. Graceful Shutdown

#### Signal Handlers
- **SIGINT Handler**: Ctrl+C triggers graceful shutdown
- **SIGTERM Handler**: Termination signal triggers graceful shutdown
- **Checkpoint Save**: Automatically saves checkpoint before exit
- **Workflow Status**: Marks workflow as CRASHED for resume

#### Shutdown Process
1. Stop accepting new work
2. Save current checkpoint
3. Mark workflow as CRASHED
4. Run complete cleanup
5. Close all connections
6. Exit cleanly

### 5. Progress Tracking

#### Progress Bar
- **Real-time Progress**: tqdm progress bar with file-by-file updates
- **Memory Indicators**: Shows current memory usage percentage
- **Batch Progress**: Tracks progress within each batch
- **Overall Progress**: Shows total files processed

#### Detailed Logging
- **Memory Statistics**: Before/after memory for each file
- **Batch Progress**: Memory stats, duration, success/failure counts
- **OOM Recovery Actions**: Logs all OOM recovery attempts
- **Checkpoint Status**: Save/validate operations logged

## Architecture

### Component Diagram

```
MemorySafeBatchProcessor
    |
    +-- MemoryManager
    |   +-- MemoryPredictor
    |   +-- ServiceCleanupProtocol
    |   +-- FileMemoryStats
    |   +-- MemoryPressureStatus
    |
    +-- CheckpointManager
    |   +-- CheckpointValidator
    |   +-- WorkflowStateStore
    |   +-- ResumeState
    |
    +-- ErrorClassifier
    |   +-- ErrorCategory
    |   +-- RetryStrategy
    |
    +-- ProgressTracker
    +-- GracefulShutdown
```

### Processing Flow

```
1. Initialize
   - Create MemoryManager with thresholds
   - Create CheckpointManager
   - Setup signal handlers
   - Initialize services

2. Prepare
   - Scan source directory for files
   - Check for existing workflow (if --resume)
   - Validate checkpoint (if resuming)
   - Create new workflow

3. Process Batches
   For each batch:
   a. Pre-allocation check
      - Predict memory usage
      - Verify sufficient memory
      - Reduce batch size if needed

   b. Process files
      - Track file start memory
      - Process with retry logic
      - Track file end memory
      - Update checkpoint

   c. Monitor
      - Background memory watchdog
      - Progress bar updates
      - Check for shutdown signals

   d. Cleanup
      - Unload all models
      - Clear GPU cache
      - Run garbage collection
      - Save batch checkpoint

4. Complete
   - Save final checkpoint
   - Mark workflow as COMPLETED
   - Generate summary report
   - Cleanup all resources
```

## Usage

### Basic Usage

```bash
# Normal run with defaults
python scripts/run_memory_safe_batch.py --source ref/call/

# Custom batch size
python scripts/run_memory_safe_batch.py --source ref/call/ --batch-size 10

# Resume from checkpoint
python scripts/run_memory_safe_batch.py --source ref/call/ --resume
```

### Advanced Usage

```bash
# Custom memory thresholds
python scripts/run_memory_safe_batch.py \
    --source ref/call/ \
    --batch-size 5 \
    --memory-threshold 80 \
    --gpu-threshold 85

# Disable GPU (CPU-only processing)
python scripts/run_memory_safe_batch.py \
    --source ref/call/ \
    --no-gpu \
    --batch-size 3

# Maximum retries and custom checkpoint directory
python scripts/run_memory_safe_batch.py \
    --source ref/call/ \
    --batch-size 5 \
    --max-retries 5 \
    --checkpoint-dir /tmp/checkpoints

# Resume with custom output
python scripts/run_memory_safe_batch.py \
    --source ref/call/ \
    --resume \
    --output logs/batch_result.json
```

### Command-Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--source` | `-s` | `ref/call` | Directory containing files to process |
| `--batch-size` | `-b` | `5` | Number of files per batch |
| `--memory-threshold` | | `85` | System memory threshold percentage |
| `--gpu-threshold` | | `90` | GPU memory threshold percentage |
| `--gpu/--no-gpu` | | `gpu` | Enable/disable GPU acceleration |
| `--max-retries` | | `3` | Maximum retry attempts per file |
| `--checkpoint-dir` | | `data/checkpoints` | Directory for checkpoint storage |
| `--resume` | `-r` | `False` | Resume from last checkpoint |
| `--output` | `-o` | `logs/memory_safe_batch_result.json` | Output JSON file |

## Memory Management Details

### Pre-Allocation Check

Before starting each batch, the system:

1. **Calculates Memory Requirements**
   - Current memory usage
   - Estimated per-file memory (default: 10MB)
   - Safety margin (30% buffer)

2. **Predicts Memory Pressure**
   - Total memory = current + (batch_size × per_file × safety_margin)
   - Percentage = (total_memory / system_memory) × 100

3. **Takes Action**
   - **< 85%**: Safe to proceed
   - **85-95%**: Warning logged, proceed with caution
   - **> 95%**: Reduce batch size or abort

### Per-File Tracking

Each file processing operation tracks:

```python
FileMemoryStats:
    - file_path: str
    - file_size_mb: float
    - start_memory_mb: float
    - end_memory_mb: float
    - peak_memory_mb: float
    - delta_mb: float
    - processing_time_sec: float
    - success: bool
    - timestamp: datetime
    - stage: str
```

### Cleanup Process

Between batches, the system:

1. **Unloads Services**
   - STT service models
   - Forensic service models
   - SER model cache
   - All registered services

2. **Clears GPU Memory**
   - PyTorch CUDA cache
   - IPC memory (if available)
   - GPU synchronization

3. **Garbage Collection**
   - Generation 2 (oldest)
   - Generation 1 (middle)
   - Generation 0 (youngest)
   - Final pass

4. **Reports Statistics**
   - Services cleaned
   - Objects collected
   - Memory freed (MB)

## Checkpoint System Details

### Checkpoint Structure

```python
BatchCheckpoint:
    - batch_id: str
    - batch_index: int
    - processed_files: List[str]
    - failed_files: List[str]
    - pending_files: List[str]
    - results: Dict[str, Any]
    - timestamp: datetime
    - metadata: Dict[str, Any]
```

### Validation Checks

The validator checks:

1. **Database Integrity**
   - All required tables exist
   - Foreign key constraints enabled
   - No corruption in indexes

2. **Workflow Consistency**
   - Workflow state is valid
   - Timestamps are present
   - Counts are consistent
   - Metadata is JSON-serializable

3. **File Existence**
   - All tracked files exist
   - Completed files have results
   - Failed files have error messages

4. **Checkpoint Data**
   - Batch IDs are valid
   - Workflow IDs are valid
   - Results JSON is valid
   - File lists are present

### Resume Process

1. **Find Latest Workflow**
   - Search for CRASHED or RUNNING workflows
   - Select most recent by timestamp

2. **Validate Checkpoint**
   - Run all validation checks
   - Attempt repairs if possible
   - Fail if critical errors exist

3. **Load Resume State**
   - Get processed file list
   - Get failed file list
   - Determine remaining files
   - Get current batch number

4. **Continue Processing**
   - Filter out processed/failed files
   - Continue from current batch
   - Update workflow status to RUNNING

## Error Recovery Strategies

### Transient Errors
**Examples**: Network timeouts, temporary GPU unavailability

**Strategy**:
- Retry with exponential backoff
- Cleanup before retry
- Max 3 attempts

### Resource Errors
**Examples**: OOM, disk space issues

**Strategy**:
- Reduce batch size (for OOM)
- Aggressive cleanup
- Retry once

### Permanent Errors
**Examples**: Corrupted files, invalid format

**Strategy**:
- Log error
- Mark file as failed
- Skip to next file

### Service Errors
**Examples**: STT failure, forensic service error

**Strategy**:
- Retry with cleanup
- CPU fallback if GPU fails
- Skip after max retries

## Monitoring and Logging

### Log Levels

- **DEBUG**: Detailed operation logs
- **INFO**: Normal operation logs (progress, stats)
- **WARNING**: Warnings (memory pressure, retries)
- **ERROR**: Errors (file failures, OOM)
- **CRITICAL**: Critical failures (fatal errors)

### Key Log Messages

```
# Initialization
MemorySafeBatchProcessor initialized: source=ref/call, batch_size=5

# Pre-allocation check
Pre-allocation check: current=2048MB, available=10240MB, estimated_needed=600MB
Pre-allocation check: Sufficient memory: predicted 25% usage

# Batch processing
BATCH 1: Processing 5 files
Memory: 2048MB / 20% used
[1/5] Processing: file1.m4a
STT processing: file1.m4a
Forensic complete: risk=45.2
SUCCESS: file1.m4a

# Cleanup
Batch 1 completed. Running cleanup...
Starting complete cleanup between batches...
Cleanup complete: 3 services, 1250 GC objects, 450.2MB freed

# Batch summary
BATCH 1 SUMMARY:
  Files: 5/5 successful
  Failed: 0, Skipped: 0
  Retries: 0
  Duration: 125.3s
  Memory: 2048MB -> 1598MB (freed 450MB)
  Peak: 3102MB

# OOM recovery
OOM detected (count=1/3), reducing batch size: 5 -> 2
Emergency OOM cleanup freed 1250.5MB

# Shutdown
Received signal SIGINT (2), initiating graceful shutdown...
Graceful shutdown complete: User requested
Workflow marked as CRASHED for potential resume
```

## Performance Considerations

### Batch Size Selection

- **Small Batches (1-3 files)**:
  - Lower memory usage
  - More frequent checkpoints
  - Slower overall processing

- **Medium Batches (5-10 files)**:
  - Balanced memory usage
  - Good checkpoint frequency
  - Recommended default

- **Large Batches (15+ files)**:
  - Higher memory usage
  - Fewer checkpoints
  - Faster overall processing (if no OOM)

### Memory Thresholds

- **Conservative (80-85%)**:
  - Lower risk of OOM
  - More frequent cleanup
  - Slower processing

- **Moderate (85-90%)**:
  - Balanced risk/performance
  - Recommended default

- **Aggressive (90-95%)**:
  - Higher risk of OOM
  - Less frequent cleanup
  - Faster processing (if successful)

### GPU vs CPU

- **GPU Processing**:
  - 5-10x faster for STT
  - Higher memory usage
  - Recommended for production

- **CPU Processing**:
  - Slower but more stable
  - Lower memory usage
  - Good fallback option

## Troubleshooting

### Out of Memory Errors

**Symptoms**:
- Process killed with OOM
- CUDA out of memory errors
- System swap usage high

**Solutions**:
1. Reduce batch size: `--batch-size 3`
2. Lower memory threshold: `--memory-threshold 80`
3. Disable GPU: `--no-gpu`
4. Close other applications

### Checkpoint Validation Failures

**Symptoms**:
- "Checkpoint validation failed" error
- Cannot resume from checkpoint

**Solutions**:
1. Run without `--resume` to start fresh
2. Delete corrupted checkpoint: `rm data/checkpoints/state.db`
3. Check logs for specific validation errors

### Slow Processing

**Symptoms**:
- Processing takes too long
- High memory usage but slow progress

**Solutions**:
1. Increase batch size (if memory allows)
2. Enable GPU: `--gpu`
3. Check GPU is being used: `nvidia-smi`
4. Reduce number of retries: `--max-retries 1`

### Files Skipped on Resume

**Symptoms**:
- Files marked as "already completed"
- Files not processing on resume

**Solutions**:
1. Verify checkpoint state: Check logs
2. Force fresh start: Don't use `--resume`
3. Clear specific file state: Manual database edit

## Best Practices

### Production Use

1. **Start with Conservative Settings**
   ```bash
   --batch-size 5 --memory-threshold 85 --gpu-threshold 90
   ```

2. **Enable Logging**
   - Logs are saved to `logs/memory_safe_batch.log`
   - Check logs regularly for warnings

3. **Monitor Memory**
   - Use `htop` or `nvidia-smi` during processing
   - Watch for memory leaks

4. **Use Resume**
   - Always run with `--resume` for long jobs
   - Allows recovery from crashes

5. **Save Results**
   - Results saved to JSON by default
   - Keep for analysis and debugging

### Development Use

1. **Small Test Batches**
   ```bash
   --batch-size 2 --source ref/call/
   ```

2. **Disable GPU for Debugging**
   ```bash
   --no-gpu
   ```

3. **Verbose Logging**
   - Modify logging level in script
   - Add custom logging for debugging

4. **Test Resume**
   - Run with `--resume` after interruption
   - Verify checkpoint integrity

## API Reference

### MemorySafeBatchProcessor

#### Constructor

```python
MemorySafeBatchProcessor(
    source_dir: Path,
    batch_size: int = 5,
    memory_threshold: float = 85.0,
    gpu_threshold: float = 90.0,
    enable_gpu: bool = True,
    max_retries: int = 3,
    checkpoint_dir: str = "data/checkpoints",
    resume: bool = False,
)
```

#### Methods

- `get_files_to_process() -> List[Path]`
- `process_file(file_path: Path, stage: str) -> Tuple[bool, Optional[str], Optional[Dict]]`
- `process_batch(batch_number: int, files: List[Path]) -> BatchResult`
- `cleanup_between_batches() -> Dict[str, Any]`
- `handle_oom(exception: Exception) -> Optional[int]`
- `graceful_shutdown(reason: str) -> None`
- `check_pre_allocation(batch_size: int, file_size_mb: float) -> Tuple[bool, str]`
- `validate_checkpoint_before_resume() -> ValidationResult`
- `process_all_batches() -> ProcessingResult`
- `print_summary(result: ProcessingResult) -> None`

### Data Classes

#### BatchResult

```python
@dataclass
class BatchResult:
    batch_number: int
    total_files: int
    successful: int
    failed: int
    skipped: int
    retries: int
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    memory_freed_mb: float
    file_stats: List[FileMemoryStats]
    errors: List[str]
```

#### ProcessingResult

```python
@dataclass
class ProcessingResult:
    total_files: int
    total_batches: int
    total_successful: int
    total_failed: int
    total_skipped: int
    total_retries: int
    total_duration_seconds: float
    batches: List[BatchResult]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    oom_recoveries: int
    total_memory_freed_mb: float
```

## Comparison with Old Script

### Memory Improvements

| Feature | Old Script | New Script |
|---------|-----------|------------|
| Memory Monitor | Basic psutil wrapper | Full MemoryManager with tracking |
| Pre-allocation Check | No | Yes, with prediction |
| Per-file Tracking | No | Yes, with detailed stats |
| Cleanup | Basic GC | Complete service cleanup |
| OOM Recovery | Basic batch reduction | Intelligent recovery with prediction |

### Checkpoint Improvements

| Feature | Old Script | New Script |
|---------|-----------|------------|
| Validation | Basic file checks | Comprehensive validation |
| Resume | Simple file filter | Smart state restoration |
| Repair | No | Automatic minor repairs |
| Integrity | No checks | Database integrity checks |

### Error Handling Improvements

| Feature | Old Script | New Script |
|---------|-----------|------------|
| Error Classification | No | Yes, with categories |
| Retry Strategy | Fixed backoff | Intelligent per-category |
| CPU Fallback | Manual | Automatic |
| Fatal Errors | Crash | Graceful shutdown |

## Future Enhancements

### Planned Features

1. **Adaptive Batch Size**
   - Automatically adjust batch size based on performance
   - Learn optimal size from history

2. **Distributed Processing**
   - Process batches across multiple machines
   - Shared checkpoint storage

3. **Real-time Monitoring**
   - Web dashboard for monitoring
   - Live progress updates

4. **Advanced Prediction**
   - ML-based memory prediction
   - Failure prediction

5. **Smart Scheduling**
   - Schedule processing during off-hours
   - Pause/resume based on system load

## Contributing

When contributing to the memory-safe batch processing system:

1. **Maintain Compatibility**
   - Keep CLI interface stable
   - Support existing checkpoint formats

2. **Add Tests**
   - Unit tests for new features
   - Integration tests for resume

3. **Update Documentation**
   - Document new options
   - Update examples

4. **Performance Testing**
   - Test with various batch sizes
   - Measure memory usage

## License

This module is part of the Voice.Man project and follows the same license.

## Support

For issues and questions:
- Check logs in `logs/memory_safe_batch.log`
- Review checkpoint state in `data/checkpoints/`
- Open an issue with full log output

## Changelog

### Version 1.0.0 (2025-01-15)
- Initial release
- MemoryManager integration
- Enhanced checkpoint validation
- OOM recovery with batch size reduction
- Signal handlers for graceful shutdown
- Progress bar with memory indicators
- Comprehensive logging
- Full documentation
