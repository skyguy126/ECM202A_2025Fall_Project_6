import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import argparse
from pathlib import Path
import sys

# ==========================================
# 1. METRIC CALCULATION
# ==========================================

def calculate_spatial_error(estimated_df, gt_df):
    """ 
    Calculates RMSE and Max Error in METERS.
    """
    if gt_df is None or estimated_df.empty:
        return np.nan, np.nan

    # Clean Data
    est_clean = estimated_df[
        (estimated_df['est_x'] != 0) & (estimated_df['est_y'] != 0)
    ].copy()
    
    if est_clean.empty: return np.nan, np.nan

    gt_points = gt_df[['x', 'y']].values
    tree = cKDTree(gt_points)
    est_points = est_clean[['est_x', 'est_y']].values
    
    distances, _ = tree.query(est_points)
    
    # Filter 'ghost' points
    valid_distances = distances[distances < 5000] 
    
    if len(valid_distances) == 0: return np.nan, np.nan

    rmse = np.sqrt(np.mean(valid_distances**2))
    max_err = np.max(valid_distances)
    
    return rmse, max_err

def calculate_path_length(gt_df):
    """
    Calculates the total distance (in meters) the car actually drove.
    """
    if gt_df is None or len(gt_df) < 2:
        return np.nan
        
    # Calculate distance between consecutive points
    diffs = gt_df[['x', 'y']].diff().dropna()
    segment_lengths = np.sqrt(diffs['x']**2 + diffs['y']**2)
    
    return segment_lengths.sum()

# ==========================================
# 2. DATA PROCESSING
# ==========================================

def load_ground_truth(gt_folder, car_id):
    if not gt_folder.exists(): return None
    candidates = [f"vehicle_{car_id}_positions.txt", f"{car_id}.txt", f"vehicle_{car_id}.txt"]
    gt_path = None
    for c in candidates:
        if (gt_folder / c).exists():
            gt_path = gt_folder / c; break
    if gt_path is None:
        for f in gt_folder.glob("*.txt"):
            if f.stem == str(car_id) or f.stem.endswith(f"_{car_id}"):
                gt_path = f; break
    if gt_path:
        try:
            df = pd.read_csv(gt_path, header=None, names=['x', 'y'])
            return df[(df['x'] != 0) | (df['y'] != 0)]
        except: pass
    return None

def process_scenario(scenario_path):
    scenario_name = scenario_path.name
    kf_path = scenario_path / 'final_trajectory.csv'
    graph_path = scenario_path / 'graph_trajectory.csv'
    gt_dir = scenario_path / 'ground_truth'

    if not kf_path.exists() or not graph_path.exists(): return

    # Load Raw Data
    df_kf_raw = pd.read_csv(kf_path)
    df_graph = pd.read_csv(graph_path)

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
        
        # Calculate Total Path Length
        path_len = calculate_path_length(gt_df)
        
        kf_car_all = df_kf_raw[df_kf_raw['car_id'] == cid].sort_values('timestamp')
        graph_car = df_graph[df_graph['car_id'] == cid].sort_values('timestamp')

        # Filter Kalman to match Graph timestamps
        if not graph_car.empty and not kf_car_all.empty:
            kf_car_filtered = pd.merge_asof(
                graph_car[['timestamp']], 
                kf_car_all, 
                on='timestamp', 
                direction='nearest', 
                tolerance=0.05 
            ).dropna()
        else:
            kf_car_filtered = pd.DataFrame() 

        # Calculate Metrics
        kf_rmse, kf_max = calculate_spatial_error(kf_car_filtered, gt_df)
        gr_rmse, gr_max = calculate_spatial_error(graph_car, gt_df)

        # Calculate Percentage Error
        if path_len and path_len > 0:
            kf_pct = (kf_rmse / path_len) * 100 if pd.notna(kf_rmse) else np.nan
            gr_pct = (gr_rmse / path_len) * 100 if pd.notna(gr_rmse) else np.nan
            path_len_str = f"{path_len:.2f} m"
        else:
            kf_pct, gr_pct = np.nan, np.nan
            path_len_str = "N/A"

        # Print
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