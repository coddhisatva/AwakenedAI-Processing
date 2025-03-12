#!/usr/bin/env python3
"""
Metrics Dashboard - Analyze and display metrics from pipeline runs.
This tool provides a dashboard for viewing and comparing pipeline performance metrics.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_metrics_files(metrics_dir: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Load metrics files from the specified directory.
    
    Args:
        metrics_dir: Directory containing metrics files
        limit: Maximum number of files to load (most recent first)
        
    Returns:
        List of metrics dictionaries
    """
    metrics_dir_path = Path(metrics_dir)
    if not metrics_dir_path.exists():
        logger.error(f"Metrics directory {metrics_dir} does not exist")
        return []
    
    # Find all metrics files
    metrics_files = list(metrics_dir_path.glob("metrics_*.json"))
    
    # Sort by timestamp (newest first)
    metrics_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Apply limit
    metrics_files = metrics_files[:limit]
    
    # Load metrics
    metrics_list = []
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
                # Add filename for reference
                metrics["__filename"] = file_path.name
                # Add timestamp from filename or file modified time
                try:
                    timestamp_str = file_path.stem.split('_', 1)[1]
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                except (IndexError, ValueError):
                    # Fall back to file modified time
                    timestamp = datetime.fromtimestamp(os.path.getmtime(file_path))
                metrics["__timestamp"] = timestamp.isoformat()
                metrics_list.append(metrics)
        except Exception as e:
            logger.warning(f"Error loading metrics file {file_path}: {e}")
    
    logger.info(f"Loaded {len(metrics_list)} metrics files")
    return metrics_list

def display_summary(metrics_list: List[Dict[str, Any]]) -> None:
    """
    Display a summary of the loaded metrics.
    
    Args:
        metrics_list: List of metrics dictionaries
    """
    if not metrics_list:
        logger.warning("No metrics files to summarize")
        return
    
    print("\n" + "="*80)
    print("PIPELINE METRICS SUMMARY")
    print("="*80)
    
    # Display summary of most recent run
    latest_metrics = metrics_list[0]
    print(f"\nLatest Run: {latest_metrics['__timestamp']}")
    print(f"  Total processing time: {str(timedelta(seconds=latest_metrics['total_time']))}")
    print(f"  Total files: {latest_metrics['total_files']}")
    print(f"  Successfully processed: {latest_metrics['successful_files']}")
    print(f"  Total chunks created: {latest_metrics['total_chunks']}")
    
    # Phase time distribution
    print("\nPhase Time Distribution (Latest Run):")
    phases = latest_metrics['phases']
    total_phase_time = sum(phases[phase]['time'] for phase in phases)
    
    for phase in phases:
        phase_time = phases[phase]['time']
        percentage = (phase_time / total_phase_time) * 100 if total_phase_time > 0 else 0
        print(f"  {phase.capitalize()}: {percentage:.1f}% ({str(timedelta(seconds=phase_time))})")
    
    # Performance trends
    if len(metrics_list) > 1:
        print("\nPerformance Trends:")
        
        # Extract timestamps and total times for comparison
        timestamps = []
        total_times = []
        successful_files = []
        total_chunks = []
        
        for metrics in metrics_list:
            try:
                timestamp = datetime.fromisoformat(metrics["__timestamp"])
                timestamps.append(timestamp)
                total_times.append(metrics.get("total_time", 0))
                successful_files.append(metrics.get("successful_files", 0))
                total_chunks.append(metrics.get("total_chunks", 0))
            except (KeyError, ValueError) as e:
                logger.warning(f"Error processing metrics entry: {e}")
        
        # Calculate performance changes
        if len(total_times) >= 2:
            time_change = ((total_times[0] - total_times[-1]) / total_times[-1]) * 100
            direction = "faster" if time_change > 0 else "slower"
            print(f"  Processing time: {abs(time_change):.1f}% {direction} than first run")
    
    print("\n" + "="*80)

def generate_charts(metrics_list: List[Dict[str, Any]], output_dir: Optional[str] = None) -> None:
    """
    Generate charts from metrics data.
    
    Args:
        metrics_list: List of metrics dictionaries
        output_dir: Optional directory to save charts
    """
    if not metrics_list:
        logger.warning("No metrics files to generate charts")
        return
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
    
    # Prepare data
    data = []
    for metrics in metrics_list:
        try:
            timestamp = datetime.fromisoformat(metrics["__timestamp"])
            phases = metrics["phases"]
            
            entry = {
                "timestamp": timestamp,
                "total_time": metrics.get("total_time", 0),
                "successful_files": metrics.get("successful_files", 0),
                "total_chunks": metrics.get("total_chunks", 0),
                "extraction_time": phases["extraction"]["time"],
                "chunking_time": phases["chunking"]["time"],
                "embedding_time": phases["embedding"]["time"],
                "storage_time": phases["storage"]["time"]
            }
            
            # Add throughput metrics if available
            for phase in ["extraction", "chunking", "embedding", "storage"]:
                if "throughput" in phases[phase]:
                    entry[f"{phase}_throughput"] = phases[phase]["throughput"]
            
            data.append(entry)
        except (KeyError, ValueError) as e:
            logger.warning(f"Error processing metrics entry for charts: {e}")
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(data)
    df = df.sort_values("timestamp")
    
    # Only continue if we have data
    if df.empty:
        logger.warning("No valid data for charts")
        return
    
    # Set up plotting
    plt.style.use('ggplot')
    
    # Plot 1: Phase Times
    fig, ax = plt.subplots(figsize=(10, 6))
    phase_times = df[["extraction_time", "chunking_time", "embedding_time", "storage_time"]]
    phase_times.columns = ["Extraction", "Chunking", "Embedding", "Storage"]
    phase_times.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Processing Time by Phase")
    ax.set_xlabel("Run")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticklabels([f"Run {i+1}" for i in range(len(df))])
    ax.legend(title="Phase")
    
    if output_dir:
        plt.savefig(output_path / "phase_times.png", dpi=300, bbox_inches="tight")
    
    # Plot 2: Throughput
    fig, ax = plt.subplots(figsize=(10, 6))
    throughput_cols = [col for col in df.columns if "throughput" in col]
    if throughput_cols:
        throughput = df[throughput_cols]
        throughput.columns = [col.replace("_throughput", "").capitalize() for col in throughput_cols]
        throughput.plot(ax=ax, marker='o')
        ax.set_title("Processing Throughput by Phase")
        ax.set_xlabel("Run")
        ax.set_ylabel("Items per Second")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([f"Run {i+1}" for i in range(len(df))])
        ax.legend(title="Phase")
        
        if output_dir:
            plt.savefig(output_path / "throughput.png", dpi=300, bbox_inches="tight")
    
    # Plot 3: Total Processing Time
    fig, ax = plt.subplots(figsize=(10, 6))
    df["total_time"].plot(ax=ax, marker='o')
    ax.set_title("Total Processing Time")
    ax.set_xlabel("Run")
    ax.set_ylabel("Time (seconds)")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"Run {i+1}" for i in range(len(df))])
    
    if output_dir:
        plt.savefig(output_path / "total_time.png", dpi=300, bbox_inches="tight")
    
    # Show plots if not saving
    if not output_dir:
        plt.show()
    else:
        logger.info(f"Saved charts to {output_dir}")

def main():
    """Main entry point for the metrics dashboard."""
    parser = argparse.ArgumentParser(description="Pipeline metrics dashboard")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        help="Directory containing metrics files. Default is data/metrics",
        default=os.path.join("data", "metrics")
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of metrics files to load. Default is 10",
        default=10
    )
    parser.add_argument(
        "--generate_charts",
        action="store_true",
        help="Generate charts from metrics data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save charts. Default is data/metrics/charts",
        default=None
    )
    
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.generate_charts and not args.output_dir:
        args.output_dir = os.path.join(args.metrics_dir, "charts")
    
    # Load metrics files
    metrics_list = load_metrics_files(args.metrics_dir, args.limit)
    
    # Display summary
    display_summary(metrics_list)
    
    # Generate charts if requested
    if args.generate_charts:
        try:
            import matplotlib
            import pandas
            generate_charts(metrics_list, args.output_dir)
        except ImportError:
            logger.error("Could not generate charts. Please install matplotlib and pandas: pip install matplotlib pandas")

if __name__ == "__main__":
    main() 