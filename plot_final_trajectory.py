import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import json
import sys
from pathlib import Path

# Add scripts directory to path to import util
sys.path.insert(0, str(Path('scripts')))

def load_trajectory_data(csv_path):
    """Load trajectory CSV file"""
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return None
    return pd.read_csv(csv_path)

def load_ground_truth_files(demo_path):
    """Load all ground truth txt files from demo directory"""
    ground_truth_dir = demo_path / 'ground_truth'
    ground_truth_files = {}
    if ground_truth_dir.exists():
        for txt_file in ground_truth_dir.glob('*.txt'):
            gt_data = pd.read_csv(txt_file, header=None, names=['x', 'y'])
            # Filter out zero entries (where vehicle hasn't started)
            gt_data = gt_data[(gt_data['x'] != 0) | (gt_data['y'] != 0)]
            ground_truth_files[txt_file.stem] = gt_data
    return ground_truth_files

def load_events(demo_path):
    """Load final events JSON file"""
    final_events = []
    
    final_events_path = demo_path / 'final_events.json'
    if final_events_path.exists():
        with open(final_events_path, 'r') as f:
            final_events = json.load(f)
    
    return final_events

def plot_trajectory(df, label='Estimated'):
    """Plot estimated trajectory"""
    return plt.scatter(df['est_y'], df['est_x'], s=10, alpha=0.6, c=df['timestamp'], cmap='viridis', label=label)

def plot_ground_truth(ground_truth_files):
    """Plot all ground truth trajectories"""
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    for idx, (name, gt_data) in enumerate(ground_truth_files.items()):
        color = colors[idx % len(colors)]
        plt.scatter(gt_data['y'], gt_data['x'], s=10, alpha=0.6, c=color, marker='x', label=f'Ground Truth: {name}')

def plot_events(events):
    """Plot events at camera locations with car ID labels"""
    if not events:
        return
    
    edge_events = []
    inner_events = []
    
    # Separate edge and inner events
    for event in events:
        if event['type'] == 'edge':
            edge_events.append(event)
        else:
            inner_events.append(event)
    
    # Plot edge events
    for event in edge_events:
        cam_x, cam_y = event['location']['x'], event['location']['y']
        if cam_x is not None and event['car_id']:
            plt.scatter(cam_y, cam_x, s=200, alpha=0.8, c='darkred', marker='*', label='Edge Events' if event == edge_events[0] else '')
            plt.text(cam_y, cam_x - 8, f"Cam {event['camera_id']}\n{event['car_id']}", fontsize=9, color='darkred', fontweight='bold', ha='center', va='top')
    
    # Plot inner events
    for event in inner_events:
        cam_x, cam_y = event['location']['x'], event['location']['y']
        if cam_x is not None and event['car_id']:
            plt.scatter(cam_y, cam_x, s=200, alpha=0.8, c='darkblue', marker='s', label='Inner Events' if event == inner_events[0] else '')
            plt.text(cam_y, cam_x - 8, f"Cam {event['camera_id']}\n{event['car_id']}", fontsize=9, color='darkblue', fontweight='bold', ha='center', va='top')

def finalize_plot(sc, demo_name, demo_path):
    """Add colorbar, labels, legend, and save the plot"""
    cbar = plt.colorbar(sc, ax=plt.gca(), label='Timestamp (s)')
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}s'))

    plt.xlabel('Y Position')
    plt.ylabel('X Position')
    plt.title(f'Vehicle Trajectory - {demo_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    output_path = demo_path / 'trajectory_plot.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot vehicle trajectory from CARLA simulation')
    parser.add_argument('demo_name', type=str, help='Name of the demo scenario (folder in demos/)')
    args = parser.parse_args()
    
    # Construct path to CSV file
    csv_path = Path('demos') / args.demo_name / 'final_trajectory.csv'
    demo_path = csv_path.parent
    
    # Load data
    df = load_trajectory_data(csv_path)
    if df is None:
        return 1
    
    ground_truth_files = load_ground_truth_files(demo_path)
    final_events = load_events(demo_path)
    
    # Create figure and plot
    plt.figure(figsize=(12, 8))
    sc = plot_trajectory(df)
    plot_ground_truth(ground_truth_files)
    plot_events(final_events)
    finalize_plot(sc, args.demo_name, demo_path)
    
    return 0

if __name__ == '__main__':
    exit(main())