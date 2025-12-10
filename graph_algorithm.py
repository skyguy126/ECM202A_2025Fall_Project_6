import os
import argparse
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from M202A_algorithm2 import load_json_data

# --- Configuration ---
MAX_SPEED_MPS = 40.0   # Relaxed slightly to allow for noise
MIN_SPEED_MPS = 0.5    # Minimum speed to prevent matching stationary noise far away
AVG_SPEED_MPS = 8.0   # Target speed for cost calculation
TIME_WINDOW = 30.0     # Max seconds to look forward for connections (reduces graph size)

def solve_global_tracking(edge_data, inner_data):
    """
    Solves the tracking problem as a Multi-Commodity Flow MIP.
    Forces every inner event to be explained by a car or identified as noise.
    """
    
    # --- 1. Preprocess Data ---
    # Group edge data into trips
    edge_data.sort(key=lambda x: x[0])
    car_trips = {} 
    
    for ev in edge_data:
        cid = ev[3]
        if cid not in car_trips: car_trips[cid] = []
        car_trips[cid].append(ev)

    cars = []
    for cid, events in car_trips.items():
        if len(events) >= 1:
            # Take first as start, last as end. 
            # If only 1 event, start=end (this might be an entry that never exited, or vice versa)
            start_ev = events[0]
            end_ev = events[-1]
            cars.append({
                'id': cid,
                'start': start_ev,
                'end': end_ev,
                'start_idx': -1,
                'end_idx': -1
            })
    
    print(f"Identified {len(cars)} car trips.")

    # --- 2. Build Graph Nodes ---
    inner_nodes = []
    for i, ev in enumerate(inner_data):
        inner_nodes.append({
            'type': 'inner',
            'id': i, # This will be the index in all_nodes
            'data': ev,
            'pos': np.array([ev[1], ev[2]]),
            't': ev[0]
        })
        
    node_counter = len(inner_nodes)
    all_nodes = inner_nodes[:]
    
    for car in cars:
        # Start Node
        s_ev = car['start']
        s_node = {
            'type': 'start',
            'car_id': car['id'],
            'pos': np.array([s_ev[1], s_ev[2]]),
            't': s_ev[0],
            'global_idx': node_counter
        }
        car['start_node'] = s_node
        all_nodes.append(s_node)
        node_counter += 1
        
        # End Node
        e_ev = car['end']
        e_node = {
            'type': 'end',
            'car_id': car['id'],
            'pos': np.array([e_ev[1], e_ev[2]]),
            't': e_ev[0],
            'global_idx': node_counter
        }
        car['end_node'] = e_node
        all_nodes.append(e_node)
        node_counter += 1

    # --- 3. Initialize Solver ---
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("SCIP solver not found.")
        return pd.DataFrame()

    variables = {} # Key: (u_idx, v_idx, car_id)
    
    def get_edge_cost(u, v):
        dt = v['t'] - u['t']
        if dt <= 0: return None
        
        dist = np.linalg.norm(u['pos'] - v['pos'])
        speed = dist / dt
        
        if speed > MAX_SPEED_MPS: return None
        
        # Cost: Penalize deviation from avg speed, and distance
        speed_cost = abs(speed - AVG_SPEED_MPS)
        return (speed_cost * 2) + (dist * 0.05)

    # --- 4. Build Edges ---
    print("Building graph edges...")
    
    # We add a "Dummy Car" ID = -1 to absorb noise events.
    # It can visit any inner node from any inner node, but at HIGH cost.
    # This makes the problem feasible even if points are unreachable.
    DUMMY_ID = -1 

    # For Real Cars
    for car in cars:
        cid = car['id']
        s_n = car['start_node']
        e_n = car['end_node']
        
        # 1. Start -> End
        cost = get_edge_cost(s_n, e_n)
        if cost is not None or s_n == e_n: # Allow direct connection even if time=0 (same event)
             variables[(s_n['global_idx'], e_n['global_idx'], cid)] = solver.BoolVar(f"x_{cid}_s_e")

        # Select valid inner nodes (time between start and end)
        # We assume strict temporal ordering: Start <= Inner <= End
        valid_inner = [n for n in inner_nodes if s_n['t'] < n['t'] < e_n['t']]
        
        for n in valid_inner:
            # Start -> Inner
            cost = get_edge_cost(s_n, n)
            if cost is not None:
                variables[(s_n['global_idx'], n['id'], cid)] = solver.BoolVar(f"x_{cid}_s_{n['id']}")
            
            # Inner -> End
            cost = get_edge_cost(n, e_n)
            if cost is not None:
                variables[(n['id'], e_n['global_idx'], cid)] = solver.BoolVar(f"x_{cid}_{n['id']}_e")
                
            # Inner -> Inner (Chain)
            # To allow multiple inner nodes, we connect inner nodes to each other
            for n2 in valid_inner:
                if n2['id'] == n['id']: continue
                if n2['t'] <= n['t']: continue # Must move forward in time
                if n2['t'] - n['t'] > TIME_WINDOW: continue 
                
                cost = get_edge_cost(n, n2)
                if cost is not None:
                    variables[(n['id'], n2['id'], cid)] = solver.BoolVar(f"x_{cid}_{n['id']}_{n2['id']}")

    # For Dummy Car (Noise)
    # It needs a virtual source/sink, but we can simplify by just saying
    # it can "cover" any node.
    # Actually, simpler approach: Add a slack variable for each inner node
    # "y_i = 1 if node i is noise". Cost = HIGH.
    noise_vars = {}
    for n in inner_nodes:
        noise_vars[n['id']] = solver.BoolVar(f"noise_{n['id']}")


    print(f"Created {len(variables) + len(noise_vars)} variables.")

    # --- 5. Constraints ---
    
    # C1: Every Inner Node must be covered EXACTLY ONCE
    # Sum(incoming from real cars) + Is_Noise == 1
    for n in inner_nodes:
        idx = n['id']
        incoming_vars = []
        for (u, v, c), var in variables.items():
            if v == idx:
                incoming_vars.append(var)
        
        # Add the noise variable for this node
        incoming_vars.append(noise_vars[idx])
        
        solver.Add(sum(incoming_vars) == 1)

    # C2: Flow Conservation for Real Cars
    for car in cars:
        cid = car['id']
        s_idx = car['start_node']['global_idx']
        e_idx = car['end_node']['global_idx']
        
        # Source Out = 1
        out_s = [var for (u, v, c), var in variables.items() if u == s_idx and c == cid]
        solver.Add(sum(out_s) == 1)
        
        # Sink In = 1
        in_e = [var for (u, v, c), var in variables.items() if v == e_idx and c == cid]
        solver.Add(sum(in_e) == 1)
        
        # Inner Nodes Flow Balance (In = Out)
        # Only relevant for nodes this car *could* visit
        valid_inner_indices = [n['id'] for n in inner_nodes if car['start_node']['t'] < n['t'] < car['end_node']['t']]
        
        for idx in valid_inner_indices:
            in_vars = [var for (u, v, c), var in variables.items() if v == idx and c == cid]
            out_vars = [var for (u, v, c), var in variables.items() if u == idx and c == cid]
            
            # If a car enters a node, it must leave it (unless it's the end node)
            if in_vars or out_vars:
                solver.Add(sum(in_vars) == sum(out_vars))

    # --- 6. Objective ---
    objective = solver.Objective()
    
    # Cost for real edges
    for (u_idx, v_idx, cid), var in variables.items():
        u_node = all_nodes[u_idx] if u_idx >= len(inner_nodes) else inner_nodes[u_idx]
        v_node = all_nodes[v_idx] if v_idx >= len(inner_nodes) else inner_nodes[v_idx]
        
        # Base cost from physics
        cost = get_edge_cost(u_node, v_node)
        if cost is None: cost = 0 # Should check earlier, but safety
        
        objective.SetCoefficient(var, cost)
        
    # Cost for noise (High penalty so we prefer real cars)
    NOISE_PENALTY = 1000.0
    for idx, var in noise_vars.items():
        objective.SetCoefficient(var, NOISE_PENALTY)
        
    objective.SetMinimization()

    # --- 7. Solve ---
    print("Solving MIP...")
    status = solver.Solve()

    final_log = []
    
    if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
        print(f"Solution Found! Cost: {objective.Value()}")
        
        # Extract tracks
        for car in cars:
            cid = car['id']
            
            # 1. Start Point
            final_log.append({
                'timestamp': round(car['start_node']['t'], 1),
                'car_id': cid,
                'est_x': round(car['start_node']['pos'][0], 2),
                'est_y': round(car['start_node']['pos'][1], 2),
                'source': 'start'
            })

            # 2. Follow the edges
            curr = car['start_node']['global_idx']
            visited = set()
            
            while True:
                # Find outgoing edge with flow=1
                next_node = None
                for (u, v, c), var in variables.items():
                    if u == curr and c == cid and var.solution_value() > 0.5:
                        next_node = v
                        break
                
                if next_node is None: break
                if next_node in visited: break # Loop protection
                visited.add(next_node)
                
                # Retrieve node object
                if next_node >= len(inner_nodes):
                    # It's an end node
                    n_obj = all_nodes[next_node]
                    final_log.append({
                        'timestamp': round(n_obj['t'], 1),
                        'car_id': cid,
                        'est_x': round(n_obj['pos'][0], 2),
                        'est_y': round(n_obj['pos'][1], 2),
                        'source': 'end'
                    })
                    break # Reached end
                else:
                    # It's an inner node
                    n_obj = inner_nodes[next_node]
                    final_log.append({
                        'timestamp': round(n_obj['t'], 1),
                        'car_id': cid,
                        'est_x': round(n_obj['pos'][0], 2),
                        'est_y': round(n_obj['pos'][1], 2),
                        'source': 'inner'
                    })
                    curr = next_node
                    
        # Check noise
        noise_count = 0
        for idx, var in noise_vars.items():
            if var.solution_value() > 0.5:
                noise_count += 1
        print(f"Events classified as noise/unreachable: {noise_count}")
        
    else:
        print("No solution found.")

    return pd.DataFrame(final_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=str, help="Name of demo to run")
    args = parser.parse_args()
    
    base_path = os.path.join("demos", args.scenario)
    inner_path = os.path.join(base_path, "all_inner_events.json")
    edge_path = os.path.join(base_path, "all_edge_events.json")
    out_file = os.path.join(base_path, "graph_trajectory.csv")

    inner_data = load_json_data(inner_path, is_edge_file=False)
    edge_data = load_json_data(edge_path, is_edge_file=True)

    if inner_data or edge_data:
        df = solve_global_tracking(edge_data, inner_data)
        if not df.empty:
            df.sort_values(by=['car_id', 'timestamp'], inplace=True)
            df.to_csv(out_file, index=False)
            print(f"Saved graph-optimized trajectory to {out_file}")