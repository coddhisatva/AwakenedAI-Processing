import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

# Configure logging
logger = logging.getLogger(__name__)

class PipelineMetrics:
    """
    Unified metrics framework for the RAG pipeline.
    Tracks performance metrics across all pipeline phases.
    """
    
    def __init__(self, metrics_dir: Optional[str] = None):
        """
        Initialize the metrics tracker.
        
        Args:
            metrics_dir: Optional directory to save metrics data for persistence
        """
        # Set up metrics directory for persistence
        self.metrics_dir = None
        if metrics_dir:
            self.metrics_dir = Path(metrics_dir)
            self.metrics_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize metrics structure
        self.reset()
        
        # Track start time
        self.start_time = time.time()
    
    def reset(self):
        """Reset all metrics to initial state."""
        self.stats = {
            # Overall metrics
            "total_time": 0,
            "start_timestamp": datetime.now().isoformat(),
            "total_files": 0,
            "manifest_files": 0,
            "database_files": 0,
            "new_files": 0,
            "skipped_ocr_files": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            
            # Phase-specific metrics
            "phases": {
                "extraction": {
                    "time": 0,
                    "files_processed": 0,
                    "successful_files": 0,
                    "failed_files": 0,
                    "pdf_files": 0,
                    "epub_files": 0,
                    "other_files": 0,
                    "ocr_files": 0,
                    "throughput": 0  # files/second
                },
                "chunking": {
                    "time": 0,
                    "documents_processed": 0,
                    "successful_documents": 0,
                    "failed_documents": 0,
                    "chunks_created": 0,
                    "avg_chunks_per_doc": 0,
                    "throughput": 0  # chunks/second
                },
                "embedding": {
                    "time": 0,
                    "chunks_processed": 0,
                    "successful_chunks": 0,
                    "failed_chunks": 0,
                    "tokens_processed": 0,
                    "throughput": 0,  # chunks/second
                    "token_throughput": 0,  # tokens/second
                    "api_calls": 0,
                    "api_errors": 0,
                    "total_batches": 0,
                    "avg_batch_size": 0
                },
                "storage": {
                    "time": 0,
                    "chunks_stored": 0,
                    "successful_chunks": 0,
                    "failed_chunks": 0,
                    "throughput": 0,  # chunks/second
                    "total_batches": 0,
                    "avg_batch_size": 0
                }
            },
            
            # Detailed timings
            "timings": {}
        }
    
    def update(self, phase: str, metric: str, value: Any):
        """
        Update a specific metric for a phase.
        
        Args:
            phase: Phase name (extraction, chunking, embedding, storage)
            metric: Metric name to update
            value: Value to update with (can be incremental or absolute)
        """
        if phase == "overall":
            self.stats[metric] = value
        elif phase in self.stats["phases"] and metric in self.stats["phases"][phase]:
            self.stats["phases"][phase][metric] = value
        else:
            logger.warning(f"Attempted to update unknown metric: {phase}.{metric}")
    
    def increment(self, phase: str, metric: str, value: Any = 1):
        """
        Increment a specific metric for a phase.
        
        Args:
            phase: Phase name (extraction, chunking, embedding, storage)
            metric: Metric name to increment
            value: Value to increment by (default: 1)
        """
        if phase == "overall":
            if metric in self.stats:
                self.stats[metric] += value
            else:
                logger.warning(f"Attempted to increment unknown metric: {metric}")
        elif phase in self.stats["phases"] and metric in self.stats["phases"][phase]:
            self.stats["phases"][phase][metric] += value
        else:
            logger.warning(f"Attempted to increment unknown metric: {phase}.{metric}")
    
    def add_timing(self, name: str, elapsed: float, phase: Optional[str] = None):
        """
        Add a timing measurement.
        
        Args:
            name: Name of the operation timed
            elapsed: Elapsed time in seconds
            phase: Optional phase to categorize this timing
        """
        self.stats["timings"][name] = elapsed
        
        # If a phase is specified, update the phase timing
        if phase and phase in self.stats["phases"]:
            self.stats["phases"][phase]["time"] += elapsed
    
    def calculate_derived_metrics(self):
        """Calculate metrics derived from raw counters."""
        # Overall
        self.stats["total_time"] = time.time() - self.start_time
        
        # Extraction phase
        extraction = self.stats["phases"]["extraction"]
        if extraction["time"] > 0 and extraction["files_processed"] > 0:
            extraction["throughput"] = extraction["files_processed"] / extraction["time"]
        
        # Chunking phase
        chunking = self.stats["phases"]["chunking"]
        if chunking["time"] > 0 and chunking["chunks_created"] > 0:
            chunking["throughput"] = chunking["chunks_created"] / chunking["time"]
        if chunking["documents_processed"] > 0:
            chunking["avg_chunks_per_doc"] = chunking["chunks_created"] / chunking["documents_processed"]
        
        # Embedding phase
        embedding = self.stats["phases"]["embedding"]
        if embedding["time"] > 0:
            if embedding["chunks_processed"] > 0:
                embedding["throughput"] = embedding["chunks_processed"] / embedding["time"]
            if embedding["tokens_processed"] > 0:
                embedding["token_throughput"] = embedding["tokens_processed"] / embedding["time"]
        
        # Storage phase
        storage = self.stats["phases"]["storage"]
        if storage["time"] > 0 and storage["chunks_stored"] > 0:
            storage["throughput"] = storage["chunks_stored"] / storage["time"]
    
    def log_summary(self):
        """Log a comprehensive summary of all metrics."""
        # Calculate derived metrics
        self.calculate_derived_metrics()
        
        # Log the summary
        logger.info("\n" + "="*50)
        logger.info("PIPELINE PERFORMANCE SUMMARY")
        logger.info("="*50)
        
        # Overall metrics
        logger.info("\nOVERALL METRICS:")
        logger.info(f"Total processing time: {str(timedelta(seconds=self.stats['total_time']))}")
        logger.info(f"Start time: {self.stats['start_timestamp']}")
        logger.info(f"Total files found: {self.stats['total_files']}")
        if self.stats.get("skipped_ocr_files", 0) > 0:
            logger.info(f"Files skipped due to OCR: {self.stats['skipped_ocr_files']}")
        logger.info(f"Files in manifest: {self.stats['manifest_files']}")
        logger.info(f"Files in database: {self.stats['database_files']}")
        logger.info(f"New files: {self.stats['new_files']}")
        logger.info(f"Successfully processed: {self.stats['successful_files']}")
        if self.stats.get("failed_files", 0) > 0:
            logger.info(f"Failed to process: {self.stats['failed_files']}")
        logger.info(f"Total chunks created: {self.stats['total_chunks']}")
        
        # Phase metrics
        logger.info("\nPHASE-SPECIFIC METRICS:")
        
        # Extraction
        ext = self.stats["phases"]["extraction"]
        logger.info("\n1. EXTRACTION PHASE:")
        logger.info(f"  Time: {str(timedelta(seconds=ext['time']))}")
        logger.info(f"  Files processed: {ext['files_processed']}")
        logger.info(f"  Successful files: {ext['successful_files']}")
        logger.info(f"  Failed files: {ext['failed_files']}")
        logger.info(f"  PDF files: {ext['pdf_files']}")
        logger.info(f"  EPUB files: {ext['epub_files']}")
        logger.info(f"  Other files: {ext['other_files']}")
        logger.info(f"  OCR files: {ext['ocr_files']}")
        logger.info(f"  Throughput: {ext['throughput']:.2f} files/second")
        
        # Chunking
        chunk = self.stats["phases"]["chunking"]
        logger.info("\n2. CHUNKING PHASE:")
        logger.info(f"  Time: {str(timedelta(seconds=chunk['time']))}")
        logger.info(f"  Documents processed: {chunk['documents_processed']}")
        logger.info(f"  Successful documents: {chunk['successful_documents']}")
        logger.info(f"  Failed documents: {chunk['failed_documents']}")
        logger.info(f"  Chunks created: {chunk['chunks_created']}")
        logger.info(f"  Avg chunks per document: {chunk['avg_chunks_per_doc']:.2f}")
        logger.info(f"  Throughput: {chunk['throughput']:.2f} chunks/second")
        
        # Embedding
        emb = self.stats["phases"]["embedding"]
        logger.info("\n3. EMBEDDING PHASE:")
        logger.info(f"  Time: {str(timedelta(seconds=emb['time']))}")
        logger.info(f"  Chunks processed: {emb['chunks_processed']}")
        logger.info(f"  Successful chunks: {emb['successful_chunks']}")
        logger.info(f"  Failed chunks: {emb['failed_chunks']}")
        logger.info(f"  Tokens processed: {emb['tokens_processed']}")
        logger.info(f"  API calls: {emb['api_calls']}")
        logger.info(f"  API errors: {emb['api_errors']}")
        logger.info(f"  Total batches: {emb['total_batches']}")
        logger.info(f"  Avg batch size: {emb['avg_batch_size']:.2f}")
        logger.info(f"  Throughput: {emb['throughput']:.2f} chunks/second")
        logger.info(f"  Token throughput: {emb['token_throughput']:.2f} tokens/second")
        
        # Storage
        store = self.stats["phases"]["storage"]
        logger.info("\n4. STORAGE PHASE:")
        logger.info(f"  Time: {str(timedelta(seconds=store['time']))}")
        logger.info(f"  Chunks stored: {store['chunks_stored']}")
        logger.info(f"  Successful chunks: {store['successful_chunks']}")
        logger.info(f"  Failed chunks: {store['failed_chunks']}")
        logger.info(f"  Total batches: {store['total_batches']}")
        logger.info(f"  Avg batch size: {store['avg_batch_size']:.2f}")
        logger.info(f"  Throughput: {store['throughput']:.2f} chunks/second")
        
        # Cross-phase analytics
        logger.info("\nCROSS-PHASE ANALYTICS:")
        
        # Calculate phase time percentages
        total_phase_time = sum(self.stats["phases"][phase]["time"] for phase in self.stats["phases"])
        if total_phase_time > 0:
            logger.info("Time distribution by phase:")
            for phase in self.stats["phases"]:
                phase_time = self.stats["phases"][phase]["time"]
                percentage = (phase_time / total_phase_time) * 100
                logger.info(f"  {phase.capitalize()}: {percentage:.1f}% ({str(timedelta(seconds=phase_time))})")
        
        # Other cross-phase metrics
        if self.stats["total_files"] > 0:
            logger.info(f"Overall chunks per file: {self.stats['total_chunks'] / self.stats['total_files']:.2f}")
        
        # Detailed timings
        logger.info("\nDETAILED TIMINGS:")
        for op, time_taken in self.stats["timings"].items():
            logger.info(f"  {op}: {str(timedelta(seconds=time_taken))}")
        
        logger.info("\nProcessing complete!")
    
    def save(self, filename: Optional[str] = None):
        """
        Save metrics to a file for persistence.
        
        Args:
            filename: Optional specific filename to use, otherwise generate based on timestamp
        """
        if not self.metrics_dir:
            logger.warning("Metrics directory not set. Cannot save metrics.")
            return
        
        # Calculate derived metrics
        self.calculate_derived_metrics()
        
        # Generate filename based on timestamp if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        # Save metrics to file
        output_path = self.metrics_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")
        return output_path


class PhaseTimer:
    """Timer class for tracking phase-specific operations."""
    
    def __init__(self, name: str, metrics: PipelineMetrics, phase: Optional[str] = None):
        """
        Initialize a phase timer.
        
        Args:
            name: Name of the operation being timed
            metrics: Metrics object to update
            phase: Optional phase to associate this timer with
        """
        self.name = name
        self.metrics = metrics
        self.phase = phase
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.elapsed
        
        # Update the metrics with this timing
        self.metrics.add_timing(self.name, elapsed, self.phase)
        
    @property
    def elapsed(self):
        """Return elapsed time in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time
    
    @property
    def elapsed_str(self):
        """Return formatted elapsed time string."""
        seconds = self.elapsed
        return str(timedelta(seconds=seconds))
    
    def log(self, message=None):
        """Log elapsed time with optional message."""
        msg = f"{self.name}: {self.elapsed_str}"
        if message:
            msg = f"{msg} - {message}"
        logger.info(msg) 