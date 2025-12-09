import json
import numpy as np
import pandas as pd
import os
import argparse

DT = 0.1 
CONFIDENCE_THRESHOLD = 0.2
MIN_FRAMES_TO_EXIT = 10 

# Edge camera bounds to determine exit 
EXIT_X_MIN, EXIT_X_MAX = 15.0, 50.0  
EXIT_Y_THRESHOLD = 175.0 

CAMERA_LOCATIONS = {
    4: (35.000, -200.000), 5: (30.000, 202.500),
    1: (35.000, -150.000), 2: (30.000, -50.000), 3: (30.000, 40.000),
    6: (30.000, 142.500), 7: (62.500, -2.500), 8: (67.500, -90.000),
    9: (72.500, 85.000), 10: (127.500, 0.000), 11: (132.500, -132.500),
    12: (132.500, 127.500), 13: (-12.500, -90.000), 14: (-12.500, 2.500),
    15: (-12.500, 87.500), 16: (-50.000, -42.500), 17: (-50.000, 42.500),
    18: (-87.500, 0.000), 19: (-87.500, -92.500), 20: (-87.500, 87.500),
    21: (-75.000, 145.000), 22: (-75.000, -137.500), 23: (-162.500, -92.500),
    24: (-155.000, -5.000), 25: (-160.000, 87.500), 26: (-125.000, 45.000),
    27: (-125.000, -45.000), 28: (-175.000, -137.500), 29: (-175.000, 145.000)
}

def parse_time_str(time_str):
    ''' Converts timestamp to seconds '''
    try:
        if time_str is None: return 0.0
        if isinstance(time_str, (float, int)): return float(time_str)
        s = str(time_str).strip()
        parts = s.split(':')
        if len(parts) == 2: return (float(parts[0]) * 60.0) + float(parts[1])
        return float(s)
    except: return 0.0

def is_valid_exit(location, car_age):
    ''' Checks if edge-camera event should be treated as an exit '''
    x, y = location[0], location[1]
    if car_age < MIN_FRAMES_TO_EXIT: return False
    if not (EXIT_X_MIN < x < EXIT_X_MAX): return False
    if abs(y) > EXIT_Y_THRESHOLD: return True
    return False

class Car_KF:
    ''' Kalman filter [x, y, vx, vy]. Uses separate noise models for inner/edge cameras '''
    def __init__(self, car_id, location, timestamp):
        self.car_id = car_id
        self.last_ts = timestamp
        self.age = 0 

        self.state = np.array([location[0], location[1], 0, 0], dtype=float)

        self.P = np.eye(4) * 10 
        self.F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = np.eye(4) * 0.5
        self.R_inner = np.eye(2) * 3.0 # More noise for inner camera
        self.R_edge = np.eye(2) * 0.1

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1

    def update(self, location, cam_type='inner'):
        z = np.array(location[:2])

        R = self.R_inner if cam_type == 'inner' else self.R_edge

        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + (K @ y)
        self.P = (np.eye(4) - K @ self.H) @ self.P

def identify_event_softmax(curr_state, event_loc):
    ''' Associates inner-camera event with most likely car using softmax '''
    if not curr_state: return None
    ids, costs = [], []

    for cid, car in curr_state.items():
        dist = np.linalg.norm(car.state[:2] - event_loc[:2])
        uncertainty = np.trace(car.P[:2, :2]) + 1e-6
        cost = dist / uncertainty
        ids.append(cid)
        costs.append(cost)
    
    costs = np.array(costs)
    costs = np.clip(costs, 0, 100)

    # Convert to probabilities 
    exp_scores = np.exp(-costs)
    sum_scores = np.sum(exp_scores)
    if sum_scores == 0: return None
    probs = exp_scores / sum_scores
    if probs[np.argmax(probs)] < CONFIDENCE_THRESHOLD: return None
    return ids[np.argmax(probs)]

def load_json_data(filepath, is_edge_file=False):
    data_list = []
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return []

    try:
        with open(filepath, 'r') as f:
            raw_data = json.load(f)
        
        print(f"\n--- Loading {os.path.basename(filepath)} ---")
        valid_count = 0
        
        for item in raw_data:
            # Timestamp
            ts_val = None
            for k in ['start_s', 'Enter Time', 'enter_time', 'time', 'timestamp', 'Timestamp']:
                if k in item: ts_val = item[k]; break
            ts = parse_time_str(ts_val)

            # Camera ID
            cam_id = "?"
            for k in ['camera_id', 'Cam #', 'Cam', 'cam', 'camera', 'Camera', 'id', 'ID']:
                if k in item: cam_id = item[k]; break
            
            # Position
            x, y = None, None
            pos_raw = None
            # for k in ['location', 'Position', 'position', 'pos', 'Location']:
            #     if k in item: pos_raw = item[k]; break
            
            if pos_raw:
                if isinstance(pos_raw, dict): x, y = pos_raw.get('x'), pos_raw.get('y')
                elif isinstance(pos_raw, list) and len(pos_raw) >= 2: x, y = pos_raw[0], pos_raw[1]

            if x is None:
                try:
                    if int(cam_id) in CAMERA_LOCATIONS:
                        x, y = CAMERA_LOCATIONS[int(cam_id)]
                except: pass

            if x is None or y is None: continue 

            if is_edge_file:
                cid = -1
                for k in ['Car ID', 'CarID', 'car_id', 'id']:
                    if k in item: cid = item[k]; break
                try: cid = int(cid)
                except: pass
                # [ts, x, y, car_id, cam_id]
                data_list.append([ts, x, y, cid, cam_id])
            else:
                # [ts, x, y, cam_id]
                data_list.append([ts, x, y, cam_id])
            
            valid_count += 1

        print(f"[SUCCESS] Loaded {valid_count} valid events.")
        print(data_list)
        data_list.sort(key=lambda x: x[0])
        return data_list

    except Exception as e:
        print(f"[ERROR] Parsing {filepath}: {e}")
        return []

def run_demo(edge_data, inner_data):
    CURR_STATE = {} 
    OUTPUT_LOG = [] 
    ALL_EVENTS = []  # Store all events with their details

    edge_ts = set(round(x[0], 1) for x in edge_data)
    inner_ts = set(round(x[0], 1) for x in inner_data)
    
    if not edge_ts and not inner_ts: return pd.DataFrame()

    min_time = min(min(edge_ts) if edge_ts else 99999, min(inner_ts) if inner_ts else 99999)
    max_time = max(max(edge_ts) if edge_ts else 0, max(inner_ts) if inner_ts else 0)
    
    all_timestamps = np.arange(min_time, max_time + 0.2, 0.1)
    
    print(f"\n--- Starting Simulation ({min_time}s to {max_time}s) ---")
    e_ptr, i_ptr = 0, 0
    
    for t_raw in all_timestamps:
        curr_ts = round(t_raw, 1)

        for car in CURR_STATE.values(): car.predict()

        # Edge camera
        while e_ptr < len(edge_data) and round(edge_data[e_ptr][0], 1) <= curr_ts:
            if round(edge_data[e_ptr][0], 1) < curr_ts - 0.05: e_ptr += 1; continue
            if abs(edge_data[e_ptr][0] - curr_ts) < 0.05:
                # [ts, x, y, cid, cam_id]
                event = edge_data[e_ptr]
                loc, cid, cam_id = event[1:3], event[3], event[4]
                
                # Record edge event
                ALL_EVENTS.append({
                    'type': 'edge',
                    'timestamp': round(event[0], 2),
                    'location': {'x': round(loc[0], 2), 'y': round(loc[1], 2)},
                    'car_id': cid,
                    'camera_id': cam_id
                })
                
                if cid not in CURR_STATE:
                    print(f"[{curr_ts:.1f}s] Cam {cam_id}: Car {cid} ENTERED")
                    CURR_STATE[cid] = Car_KF(cid, loc, curr_ts)
                else:
                    CURR_STATE[cid].update(loc, cam_type='edge')
                    if is_valid_exit(loc, CURR_STATE[cid].age):
                        print(f"[{curr_ts:.1f}s] Cam {cam_id}: Car {cid} EXITED")
                        del CURR_STATE[cid]
            e_ptr += 1

        # Inner camera
        while i_ptr < len(inner_data) and round(inner_data[i_ptr][0], 1) <= curr_ts:
            if round(inner_data[i_ptr][0], 1) < curr_ts - 0.05: i_ptr += 1; continue
            if abs(inner_data[i_ptr][0] - curr_ts) < 0.05:
                # [ts, x, y, cam_id]
                event = inner_data[i_ptr]
                loc, cam_id = event[1:3], event[3]
                
                # Assign to best-matching car
                matched = identify_event_softmax(CURR_STATE, loc)
                
                # Record inner event
                ALL_EVENTS.append({
                    'type': 'inner',
                    'timestamp': round(event[0], 2),
                    'location': {'x': round(loc[0], 2), 'y': round(loc[1], 2)},
                    'car_id': matched if matched else None,
                    'camera_id': cam_id
                })
                
                if matched:
                    print(f"[{curr_ts:.1f}s] Cam {cam_id}: Matched Car {matched}")
                    CURR_STATE[matched].update(loc, cam_type='inner')
            i_ptr += 1

        for car in CURR_STATE.values():
            OUTPUT_LOG.append({
                'timestamp': curr_ts,
                'car_id': car.car_id,
                'est_x': round(car.state[0], 2),
                'est_y': round(car.state[1], 2)
            })

    return pd.DataFrame(OUTPUT_LOG), ALL_EVENTS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run tracking algorithm on a demo scenario."
    )
    parser.add_argument("scenario", type=str, help="Name of demo to run")

    args = parser.parse_args()
    base_path = os.path.join("demos", args.scenario)
    inner_path = os.path.join(base_path, "all_inner_events.json")
    edge_path = os.path.join(base_path, "all_edge_events.json")

    out_file = os.path.join(base_path, "final_trajectory.csv")
    events_file = os.path.join(base_path, "final_events.json")

    inner_data = load_json_data(inner_path, is_edge_file=False)
    edge_data = load_json_data(edge_path, is_edge_file=True)

    if inner_data or edge_data:
        df, all_events = run_demo(edge_data, inner_data)
        
        # Save all events to JSON file
        if all_events:
            with open(events_file, 'w') as f:
                json.dump(all_events, f, indent=2)
            print(f"\n[DONE] Saved {len(all_events)} events to '{events_file}'")
        
        if not df.empty:
            df.to_csv(out_file, index=False)
            print(f"[DONE] Saved {len(df)} rows to '{out_file}'")
            print(df.head())
        else:
            print("\n[WARNING] No tracks generated.")
    else:
        print("\n[ERROR] No data loaded.")