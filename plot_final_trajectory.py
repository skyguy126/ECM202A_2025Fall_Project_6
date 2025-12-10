import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import json
import sys
from pathlib import Path

def load_trajectory_data(csv_path):
    """Load trajectory CSV file"""
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return None
    return pd.read_csv(csv_path)

def load_events(demo_path):
    """Load final events JSON file"""
    final_events = []
    
    final_events_path = demo_path / 'final_events.json'
    if final_events_path.exists():
        with open(final_events_path, 'r') as f:
            final_events = json.load(f)
    
    return final_events

def plot_trajectory_for_car(ax, car_data, car_id, graph_algorithm=False, label='Estimated'):
    """Plot estimated trajectory for a single car on given axes"""
    color = 'blue'
    car_data = car_data.sort_values('timestamp')# Plot scatter points colored by timestamp
    if graph_algorithm: 
        # Plot lines connecting the points for this car
        ax.plot(car_data['est_y'], car_data['est_x'], color=color, alpha=0.8, linewidth=2, zorder=1, label=f'Estimated')
        return ax.scatter(car_data['est_y'], car_data['est_x'], s=300, alpha=0.6, c=car_data['timestamp'], cmap='viridis', label=label, zorder=2)
    return ax.scatter(car_data['est_y'], car_data['est_x'], s=10, alpha=0.6, c=car_data['timestamp'], cmap='viridis', label=label, zorder=2)
    
def plot_ground_truth_for_car(ax, car_id, demo_path):
    """Load and plot ground truth trajectory for a specific car on given axes"""
    ground_truth_dir = demo_path / 'ground_truth'
    if not ground_truth_dir.exists():
        return
    
    # Look for ground truth file matching the car_id
    for txt_file in ground_truth_dir.glob('*.txt'):
        if str(car_id) in txt_file.stem:
            gt_data = pd.read_csv(txt_file, header=None, names=['x', 'y'])
            # Filter out zero entries (where vehicle hasn't started)
            gt_data = gt_data[(gt_data['x'] != 0) | (gt_data['y'] != 0)]
            ax.scatter(gt_data['y'], gt_data['x'], s=10, alpha=0.6, c='red', marker='x', label=f'Ground Truth')
            break

def plot_events_for_car(ax, car_id, events):
    """Plot events for a specific car on given axes"""
    if not events:
        return
    
    edge_events = []
    inner_events = []
    
    # Separate edge and inner events
    for event in events:
        if event.get('car_id') == car_id:
            if event['type'] == 'edge':
                edge_events.append(event)
            else:
                inner_events.append(event)
    
    # Plot edge events
    edge_plotted = False
    for event in edge_events:
        cam_x, cam_y = event['location']['x'], event['location']['y']
        if cam_x is not None:
            label = 'Edge Events' if not edge_plotted else ''
            ax.scatter(cam_y, cam_x, s=200, alpha=0.8, c='darkred', marker='*', label=label)
            ax.text(cam_y, cam_x - 8, f"Cam {event['camera_id']}\n{event['car_id']}", fontsize=9, color='darkred', fontweight='bold', ha='center', va='top')
            edge_plotted = True
    
    # Plot inner events
    inner_plotted = False
    for event in inner_events:
        cam_x, cam_y = event['location']['x'], event['location']['y']
        if cam_x is not None:
            label = 'Inner Events' if not inner_plotted else ''
            ax.scatter(cam_y, cam_x, s=200, alpha=0.8, c='darkblue', marker='s', label=label)
            ax.text(cam_y, cam_x - 8, f"Cam {event['camera_id']}\n{event['car_id']}", fontsize=9, color='darkblue', fontweight='bold', ha='center', va='top')
            inner_plotted = True

def finalize_subplots(fig, scatters, demo_name, demo_path, graph_algorithm=False):
    """Add colorbars, labels, legend, and save the plot with subplots"""
    # Add colorbar for each subplot
    for ax, sc in zip(fig.axes, scatters):
        if sc is not None:
            cbar = plt.colorbar(sc, ax=ax, label='Timestamp (s)')
            cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}s'))
        ax.set_xlabel('Y Position')
        ax.set_ylabel('X Position')
        ax.legend(fontsize=8)
        ax.set_xlim(-250, 250)  # add whitespace if needed
        ax.set_ylim(-300, 300)  # add whitespace if needed
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    fig.suptitle(f'Vehicle Trajectories - {demo_name}', fontsize=14, fontweight='bold')
    
    if graph_algorithm:
        output_path = demo_path / 'graph_alg_plot.png'
    else:
        output_path = demo_path / 'trajectory_plot.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot vehicle trajectory from CARLA simulation')
    parser.add_argument('demo_name', type=str, help='Name of the demo scenario (folder in demos/)')
    parser.add_argument("-g", "--graph", help="Use graph algorithm data", default=False, action='store_true')

    args = parser.parse_args()
    
    # Construct path to CSV file
    if args.graph:
        csv_path = Path('demos') / args.demo_name / 'graph_trajectory.csv'
    else:
        csv_path = Path('demos') / args.demo_name / 'final_trajectory.csv'
    demo_path = csv_path.parent
    
    # Load data
    df = load_trajectory_data(csv_path)
    if df is None:
        return 1
    
    final_events = load_events(demo_path)
    
    # Get unique car IDs and create subplots
    car_ids = sorted(df['car_id'].unique())
    num_cars = len(car_ids)

    cols = num_cars
    rows = (num_cars + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    # flatten to a 1D list
    axes = axes.ravel()
        
    scatters = []
    
    # Plot each car in its own subplot
    for idx, car_id in enumerate(car_ids):
        ax = axes[idx]
        car_data = df[df['car_id'] == car_id]
        
        # Plot trajectory
        sc = plot_trajectory_for_car(ax, car_data, car_id, args.graph)
        scatters.append(sc)
        
        # Plot ground truth
        plot_ground_truth_for_car(ax, car_id, demo_path)
        
        # Plot events
        if not args.graph:
            plot_events_for_car(ax, car_id, final_events)
        
        ax.set_title(f'Car {car_id}')
    
    # Hide unused subplots
    for idx in range(num_cars, len(axes)):
        axes[idx].set_visible(False)
    
    finalize_subplots(fig, scatters, args.demo_name, demo_path, args.graph)
    
if __name__ == '__main__':
    main()