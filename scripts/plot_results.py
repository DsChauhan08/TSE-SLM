import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def plot_temperature_results(json_file_path, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_path = Path(json_file_path)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    df = pd.DataFrame(data)
    
    # Extract model and dataset name from filename for title
    # E.g., Qwen_Qwen2.5-0.5B_gsm8k_results.json -> Qwen_Qwen2.5-0.5B, gsm8k
    name_parts = file_path.stem.split('_')
    if len(name_parts) >= 3:
        model_name = name_parts[0] + '/' + name_parts[1]
        dataset = name_parts[2]
    else:
        model_name = "Model"
        dataset = "Dataset"

    # Plot Accuracy vs Temperature
    plt.figure(figsize=(10, 6))
    
    # Check what key to plot. In the uploaded json, 'exact_match,flexible-extract' is constant at 0.4
    # and 'exact_match,strict-match' is 0.0. 
    # Let's dynamically find the accuracy metric.
    
    # Extract metrics from raw_results
    metrics = []
    for row in data:
        metrics.append(row['raw_results'])
        
    metrics_df = pd.DataFrame(metrics)
    
    if 'exact_match,flexible-extract' in metrics_df.columns:
        plt.plot(df['temperature'], metrics_df['exact_match,flexible-extract'], marker='o', linestyle='-', label='Flexible Extract')
    if 'exact_match,strict-match' in metrics_df.columns:
        plt.plot(df['temperature'], metrics_df['exact_match,strict-match'], marker='x', linestyle='--', label='Strict Match')
    
    if 'accuracy' in df.columns and df['accuracy'].notnull().any():
        plt.plot(df['temperature'], df['accuracy'], marker='s', linestyle='-', label='Accuracy')
        
    plt.title(f'{model_name} Performance on {dataset} vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(df['temperature'])
    
    out_img_path = Path(output_dir) / f"{file_path.stem}_plot.png"
    plt.savefig(out_img_path)
    print(f"Saved plot to {out_img_path}")
    
    # Plot Generation Time vs Temperature
    plt.figure(figsize=(10, 6))
    plt.plot(df['temperature'], df['time_seconds'], marker='o', linestyle='-', color='orange')
    plt.title(f'{model_name} Generation Time on {dataset} vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('Time (Seconds)')
    plt.grid(True, alpha=0.3)
    plt.xticks(df['temperature'])
    
    out_time_path = Path(output_dir) / f"{file_path.stem}_time_plot.png"
    plt.savefig(out_time_path)
    print(f"Saved time plot to {out_time_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot temperature results from JSON.")
    parser.add_argument("json_file", help="Path to the JSON results file.")
    parser.add_argument("--output_dir", default="../data/plots", help="Directory to save plots.")
    args = parser.parse_args()
    
    plot_temperature_results(args.json_file, args.output_dir)