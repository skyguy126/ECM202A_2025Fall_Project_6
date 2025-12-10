import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

FRAME_RATE = 20.0
FRAME_INTERVAL = 1.0 / FRAME_RATE

# ==========================================
# 1. METRIC CALCULATION (Temporal RMSE)
# ==========================================

def calculate_temporal_error(comparison_df):
    """
    Calculates RMSE and Max Error in METERS based on direct temporal alignment.
    Requires comparison_df to contain: ['est_x', 'est_y', 'true_x', 'true_y']
    """
    if comparison_df.empty:
        return np.nan, np.nan

    # 1. Calculate the Euclidean distance between Estimate and True position at the same frame/time
    diff_x = comparison_df['est_x'] - comparison_df['true_x']
    diff_y = comparison_df['est_y'] - comparison_df['true_y']
    
    # Distance = sqrt(dx^2 + dy^2)
    distances = np.sqrt(diff_x**2 + diff_y**2)
    
    # Filter 'ghost' points (> 5000m) 
    valid_distances = distances[distances < 5000] 
    
    if len(valid_distances) == 0: 
        return np.nan, np.nan

    # 2. Calculate RMSE and Max Error
    rmse = np.sqrt(np.mean(valid_distances**2))
    max_err = np.max(valid_distances)
    
    return rmse, max_err

def load_ground_truth(gt_folder, car_id):
    """
    Loads Ground Truth, creates a frame_id column based on row index, and applies a -1 frame offset.
    """
    if not gt_folder.exists(): return None
    candidates = [f"vehicle_{car_id}_positions.txt", f"{car_id}.txt", f"vehicle_{car_id}.txt"]
    gt_path = None
    
    # Search for the correct GT file
    for c in candidates:
        if (gt_folder / c).exists():
            gt_path = gt_folder / c; break
    if gt_path is None:
        for f in gt_folder.glob("*.txt"):
            if f.stem == str(car_id) or f.stem.endswith(f"_{car_id}"):
                gt_path = f; break
    
    if gt_path:
        try:
            # Load file: no header, only X and Y columns
            df = pd.read_csv(gt_path, header=None, names=['x', 'y'])
            
            # Create a Frame ID (0-based index)
            df['frame_id'] = df.index.values.astype(int)
            
            # Apply the -1 frame offset 
            df['frame_id'] = df['frame_id'] - 1 
            
            # Filter the row that would now have frame_id = -1
            df = df[df['frame_id'] >= 0].copy()
            
            # Rename columns to avoid clashes after merge
            df = df.rename(columns={'x': 'true_x', 'y': 'true_y'})
            return df[(df['true_x'] != 0) | (df['true_y'] != 0)]
        except Exception as e:
            print(f"Error loading GT file {gt_path}: {e}")
            return None
    return None

def calculate_path_length(gt_df):
    """
    Calculates the total distance (in meters) the car actually drove using true coordinates.
    """
    if gt_df is None or len(gt_df) < 2:
        return np.nan
        
    # Uses true coordinates to calculate path length
    diffs = gt_df[['true_x', 'true_y']].diff().dropna()
    segment_lengths = np.sqrt(diffs['true_x']**2 + diffs['true_y']**2)
    
    total_length = segment_lengths.sum()
    return total_length if total_length > 0 else 1.0

def print_sync_sample(kf_df, gr_df, label, n_samples=5):
    """Prints a sample of synchronized data for visualization."""
    
    sample_df = gr_df.sample(n=min(n_samples, len(gr_df)), random_state=42).sort_values('frame_id')

    print(f"\n--- SAMPLE SYNCHRONIZATION FOR CAR {label} ---")
    print(f"Total synced samples: {len(gr_df)} (Graph) / {len(kf_df)} (Kalman)")
    
    print("\n{:^6} | {:^8} | {:^18} | {:^18} | {:^18}".format(
        "Frame", "Time (s)", "Ground Truth (GT)", "Kalman (Est)", "Graph (Est)"))
    print("-" * 85)

    for index, gr_row in sample_df.iterrows():
        frame_id = gr_row['frame_id']
        
        # Get Kalman Row at the same frame_id
        kf_row = kf_df[kf_df['frame_id'] == frame_id].iloc[0] if not kf_df[kf_df['frame_id'] == frame_id].empty else None

        # Calculate instantaneous errors
        kf_err = np.sqrt((kf_row['est_x'] - gr_row['true_x'])**2 + (kf_row['est_y'] - gr_row['true_y'])**2) if kf_row is not None else np.nan
        gr_err = np.sqrt((gr_row['est_x'] - gr_row['true_x'])**2 + (gr_row['est_y'] - gr_row['true_y'])**2)
        
        # Format strings
        gt_pos = f"({gr_row['true_x']:.1f}, {gr_row['true_y']:.1f})"
        gr_pos = f"({gr_row['est_x']:.1f}, {gr_row['est_y']:.1f}) [E:{gr_err:.2f}]"
        kf_pos = f"({kf_row['est_x']:.1f}, {kf_row['est_y']:.1f}) [E:{kf_err:.2f}]" if kf_row is not None else "N/A"
        
        print(f"{frame_id:^6} | {gr_row['timestamp']:.6f} | {gt_pos:^18} | {kf_pos:^18} | {gr_pos:^18}")

    print("-" * 85)


def process_scenario(scenario_path):
    scenario_name = scenario_path.name
    kf_path = scenario_path / 'final_trajectory.csv'
    graph_path = scenario_path / 'graph_trajectory.csv'
    gt_dir = scenario_path / 'ground_truth'

    if not kf_path.exists() or not graph_path.exists(): return

    # Load Raw Data and rename columns
    df_kf_raw = pd.read_csv(kf_path).sort_values('timestamp')
    df_graph = pd.read_csv(graph_path).sort_values('timestamp')

    df_kf_raw = df_kf_raw.rename(columns={'x': 'est_x', 'y': 'est_y'})
    df_graph = df_graph.rename(columns={'x': 'est_x', 'y': 'est_y'})

    # Create Frame ID for ESTIMATED data using 20 FPS
    df_kf_raw['frame_id'] = (df_kf_raw['timestamp'] * FRAME_RATE).round(0).astype(int)
    df_graph['frame_id'] = (df_graph['timestamp'] * FRAME_RATE).round(0).astype(int)
    
    # Identify common cars
    cars_kf = set(df_kf_raw['car_id'].unique()) if 'car_id' in df_kf_raw else set()
    cars_graph = set(df_graph['car_id'].unique()) if 'car_id' in df_graph else set()
    all_cars = sorted(cars_kf | cars_graph)

    if not all_cars: return

    print(f"\n{'#'*80}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'#'*80}")
    print(f"{'Car ID':<8} | {'Metric':<20} | {'Kalman (Filtered)':<22} | {'Graph (Batch)':<22}")
    print("-" * 80)

    for cid in all_cars:
        gt_df = load_ground_truth(gt_dir, cid)
        
        if gt_df is None or gt_df.empty:
             print(f"{cid:<8} | {'Error: GT Missing':<20} | {'N/A':<22} | {'N/A':<22}"); continue
        
        path_len = calculate_path_length(gt_df)

        kf_car_all = df_kf_raw[df_kf_raw['car_id'] == cid].copy()
        graph_car = df_graph[df_graph['car_id'] == cid].copy()

        #  Filter Kalman Data to Graph Timestamps 
        kf_filtered = pd.merge_asof(
            graph_car[['frame_id']],     # Use Graph Frame IDs as the basis
            kf_car_all,                  # Filter the dense Kalman data
            on='frame_id',
            direction='nearest',
            tolerance=1 # Match frames within 1 frame
        ).dropna()
        
        #  Align Kalman to Ground Truth Frame ID
        kf_synced_gt = pd.merge(
            kf_filtered.sort_values('frame_id'),
            gt_df.sort_values('frame_id'),
            on='frame_id',
            how='inner'
        ).dropna(subset=['true_x', 'true_y', 'est_x', 'est_y'])
        
        # Align Graph Data to Ground Truth Frame ID 
        graph_synced_gt = pd.merge(
            graph_car.sort_values('frame_id'),
            gt_df.sort_values('frame_id'),
            on='frame_id',
            how='inner'
        ).dropna(subset=['true_x', 'true_y', 'est_x', 'est_y'])
        
        print_sync_sample(kf_synced_gt, graph_synced_gt, cid)
        
        kf_rmse, kf_max = calculate_temporal_error(kf_synced_gt)
        gr_rmse, gr_max = calculate_temporal_error(graph_synced_gt)

        # Calculate Percentage Error
        if path_len and path_len > 0:
            kf_pct = (kf_rmse / path_len) * 100 if pd.notna(kf_rmse) else np.nan
            gr_pct = (gr_rmse / path_len) * 100 if pd.notna(gr_rmse) else np.nan
            path_len_str = f"{path_len:.2f} m"
        else:
            kf_pct, gr_pct = np.nan, np.nan
            path_len_str = "N/A"

        print(f"{cid:<8} | {'Total Path Length':<20} | {path_len_str:<22} | {path_len_str:<22}")
        print(f"{'':<8} | {'RMSE (Meters)':<20} | {kf_rmse:<22.4f} | {gr_rmse:<22.4f}")
        print(f"{'':<8} | {'Error (%)':<20} | {kf_pct:<21.2f}% | {gr_pct:<21.2f}%")
        print(f"{'':<8} | {'Max Drift (Meters)':<20} | {kf_max:<22.4f} | {gr_max:<22.4f}")
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser()
    default_demos = r"C:\Users\fasts\Downloads\ECM202A_2025Fall_Project_6\demos"
    parser.add_argument("demos_path", nargs='?', default=default_demos)
    args = parser.parse_args()
    root_path = Path(args.demos_path)

    if not root_path.exists():
        print(f"[ERROR] Path not found: {root_path}")
        return

    print(f"Scanning for scenarios in: {root_path}...\n")
    
    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    for folder in sorted(subdirs):
        process_scenario(folder)

if __name__ == "__main__":
    main()