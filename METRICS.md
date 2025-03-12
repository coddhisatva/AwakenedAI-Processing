# Awakened AI - Performance Metrics System

This document provides detailed information about the unified performance metrics system implemented in the Awakened AI Processing Pipeline.

## Overview

The performance metrics system tracks, analyzes, and visualizes efficiency metrics across all pipeline stages. It provides valuable insights for optimization, troubleshooting, and monitoring production performance.

## Core Components

### 1. PipelineMetrics Class

Located in `src/pipeline/metrics.py`, this class forms the foundation of the metrics system:

```python
from src.pipeline.metrics import PipelineMetrics, PhaseTimer

# Initialize with optional persistence directory
metrics = PipelineMetrics(metrics_dir="data/metrics")
```

Key methods:

| Method | Description |
|--------|-------------|
| `reset()` | Reset all metrics to initial state |
| `update(phase, metric, value)` | Update a specific metric for a phase |
| `increment(phase, metric, value=1)` | Increment a metric for a phase |
| `add_timing(name, elapsed, phase=None)` | Add a timing measurement |
| `calculate_derived_metrics()` | Calculate metrics derived from raw counters |
| `log_summary()` | Log comprehensive metrics summary |
| `save(filename=None)` | Save metrics to a JSON file |

### 2. PhaseTimer Class

A context manager for timing operations:

```python
# Basic usage
with PhaseTimer("Operation name", metrics, phase="extraction"):
    # Code to time goes here
    process_document()
    
# Accessing elapsed time during execution
with PhaseTimer("Long operation", metrics, phase="chunking") as timer:
    process_part1()
    logger.info(f"Part 1 completed in {timer.elapsed_str}")
    process_part2()
```

### 3. Metrics Dashboard

Located in `tools/metrics_dashboard.py`, this tool provides visualization and analysis:

```bash
# Basic usage
python tools/metrics_dashboard.py

# Generate visual charts
python tools/metrics_dashboard.py --generate_charts

# Look at a specific number of recent runs
python tools/metrics_dashboard.py --limit 5

# Save charts to a custom directory
python tools/metrics_dashboard.py --generate_charts --output_dir reports/metrics
```

## Metrics Structure

The metrics framework organizes data hierarchically:

```
stats
├── total_time
├── start_timestamp
├── total_files
├── manifest_files
├── database_files
├── new_files
├── skipped_ocr_files
├── successful_files
├── failed_files
├── total_chunks
├── phases
│   ├── extraction
│   │   ├── time
│   │   ├── files_processed
│   │   ├── successful_files
│   │   ├── failed_files
│   │   ├── pdf_files
│   │   ├── epub_files
│   │   ├── other_files
│   │   ├── ocr_files
│   │   └── throughput
│   ├── chunking
│   │   ├── time
│   │   ├── documents_processed
│   │   ├── successful_documents
│   │   ├── failed_documents
│   │   ├── chunks_created
│   │   ├── avg_chunks_per_doc
│   │   └── throughput
│   ├── embedding
│   │   ├── time
│   │   ├── chunks_processed
│   │   ├── successful_chunks
│   │   ├── failed_chunks
│   │   ├── tokens_processed
│   │   ├── throughput
│   │   ├── token_throughput
│   │   ├── api_calls
│   │   ├── api_errors
│   │   ├── total_batches
│   │   └── avg_batch_size
│   └── storage
│       ├── time
│       ├── chunks_stored
│       ├── successful_chunks
│       ├── failed_chunks
│       ├── throughput
│       ├── total_batches
│       └── avg_batch_size
└── timings
    ├── operation_name_1
    ├── operation_name_2
    └── ...
```

## Integrating with Pipeline Components

### RAGPipeline Integration

The main pipeline initializes and coordinates metrics:

```python
# RAGPipeline initialization
self.metrics = PipelineMetrics(metrics_dir=metrics_dir)

# Usage throughout the pipeline
self.metrics.increment("overall", "total_files", len(files))
```

### Component Integration

Components accept a unified_metrics parameter:

```python
# Component initialization
chunker = SemanticChunker(
    processed_dir="data/processed",
    chunks_dir="data/chunks",
    unified_metrics=self.metrics  # Pass the metrics object
)

# Usage within components
if self.unified_metrics:
    self.unified_metrics.increment("chunking", "chunks_created", num_chunks)
```

## Example Metrics Output

When calling `metrics.log_summary()`, you'll see output like:

```
==================================================
PIPELINE PERFORMANCE SUMMARY
==================================================

OVERALL METRICS:
Total processing time: 0:05:23.456789
Start time: 2023-06-01T12:34:56
Total files found: 20
Files in manifest: 15
Files in database: 12
New files: 5
Successfully processed: 5
Total chunks created: 150

PHASE-SPECIFIC METRICS:

1. EXTRACTION PHASE:
  Time: 0:01:12.345678
  Files processed: 5
  Successful files: 5
  Failed files: 0
  PDF files: 3
  EPUB files: 2
  Other files: 0
  OCR files: 0
  Throughput: 0.07 files/second

... (similar sections for other phases) ...

CROSS-PHASE ANALYTICS:
Time distribution by phase:
  Extraction: 22.5% (0:01:12.345678)
  Chunking: 15.3% (0:00:49.234567)
  Embedding: 45.2% (0:02:25.789012)
  Storage: 17.0% (0:00:55.123456)

Overall chunks per file: 30.00

DETAILED TIMINGS:
  PDF extraction: 0:00:45.123456
  EPUB extraction: 0:00:27.234567
  ...
```

## Metrics Files

Metrics are saved as JSON files in the specified metrics directory:

```
data/metrics/
├── metrics_20230601_123456.json
├── metrics_20230602_134567.json
└── ...
```

The JSON structure matches the metrics structure described above.

## Best Practices

### When Adding New Components

1. Accept a `unified_metrics` parameter in the constructor
2. Use try/except when importing to allow standalone use:
   ```python
   try:
       from src.pipeline.metrics import PipelineMetrics, PhaseTimer
   except ImportError:
       # For when component is used standalone
       PipelineMetrics = None
       PhaseTimer = None
   ```
3. Check if metrics are available before using:
   ```python
   if self.unified_metrics:
       self.unified_metrics.increment("phase_name", "metric_name", value)
   ```
4. Use PhaseTimer for significant operations:
   ```python
   timer_context = (
       PhaseTimer("Operation name", self.unified_metrics, phase="phase_name")
       if self.unified_metrics and PhaseTimer
       else None
   )
   
   if timer_context:
       timer_context.__enter__()
   try:
       # Operation code
   finally:
       if timer_context:
           timer_context.__exit__(None, None, None)
   ```

### For Production Monitoring

1. Regularly analyze metrics using the dashboard tool
2. Establish performance baselines
3. Monitor trends over time
4. Set up alerts for unexpected performance changes
5. Consider implementing automated monitoring
6. Archive metrics files with a defined retention policy

## Advanced Usage

### Custom Metrics

To add custom metrics specific to your component:

1. Initialize component-specific metrics in the relevant phase
   ```python
   # In RAGPipeline.__init__
   self.metrics.stats["phases"]["my_component"] = {
       "my_custom_metric": 0,
       # other metrics...
   }
   ```

2. Update these metrics during processing
   ```python
   self.metrics.increment("my_component", "my_custom_metric", value)
   ```

### Advanced Analysis

For more advanced analysis than provided by the dashboard:

1. Load metrics files programmatically:
   ```python
   import json
   from pathlib import Path
   
   metrics_files = list(Path("data/metrics").glob("metrics_*.json"))
   metrics_data = []
   
   for file_path in metrics_files:
       with open(file_path, 'r') as f:
           metrics_data.append(json.load(f))
   ```

2. Use pandas for detailed analysis:
   ```python
   import pandas as pd
   
   # Extract specific metrics into a DataFrame
   df = pd.DataFrame([
       {
           "timestamp": m.get("start_timestamp"),
           "total_time": m.get("total_time"),
           "embedding_time": m.get("phases", {}).get("embedding", {}).get("time"),
           # other metrics...
       }
       for m in metrics_data
   ])
   
   # Analyze correlations
   correlation = df.corr()
   print(correlation)
   
   # Generate custom visualizations
   import matplotlib.pyplot as plt
   plt.figure(figsize=(10, 6))
   plt.scatter(df["total_chunks"], df["total_time"])
   plt.xlabel("Total Chunks")
   plt.ylabel("Total Processing Time (s)")
   plt.title("Processing Time vs. Chunk Count")
   plt.savefig("custom_analysis.png")
   ```

## Troubleshooting

### Common Issues

1. **Missing Metrics**: Check that unified_metrics is properly passed to all components
2. **Zero Throughput**: Ensure timers are properly closed (especially in error paths)
3. **Dashboard Errors**: Confirm matplotlib and pandas are installed
4. **Inconsistent Metrics**: Look for timer usage without proper try/finally blocks

### Debugging Metrics

1. Enable debug logging to see detailed metrics operations:
   ```python
   import logging
   logging.getLogger("src.pipeline.metrics").setLevel(logging.DEBUG)
   ```

2. Manually inspect saved metrics files for anomalies

## Future Enhancements

Potential improvements to the metrics system:

1. Real-time monitoring dashboard
2. Integration with monitoring services
3. Anomaly detection
4. Resource usage tracking (memory, CPU)
5. Cost forecasting based on API usage
6. Custom alert thresholds 