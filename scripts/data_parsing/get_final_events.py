import os
import json
from glob import glob
import argparse

FPS = 20

# Example camera config
# Camera configurations
CAMERA_CONFIGS = [
    # Visible cameras
    {"id": 4, "pos": (35.000, -210.000, 7.500), "rot": (-28.00, 86.00, 0.00)},
    {"id": 5, "pos": (27.500, 212.500, 7.500), "rot": (-28.00, 268.00, 0.00)},

    # Encrypted cameras
    {"id": 1, "pos": (35.000, -150.000, 17.500), "rot": (-90.00, 0.00, 0.00)},
    {"id": 2, "pos": (30.000, -50.000, 20.000), "rot": (-90.00, 2.00, 0.00)},
    {"id": 3, "pos": (30.000, 40.000, 20.000), "rot": (-90.00, 0.00, 0.00)},
    {"id": 6, "pos": (30.000, 142.500, 15.000), "rot": (-90.00, 0.00, 0.00)},
    {"id": 7, "pos": (62.500, -2.500, 17.500), "rot": (-90.00, 0.00, 0.00)},
    {"id": 8, "pos": (67.500, -90.000, 20.000), "rot": (-90.00, 16.00, 0.00)},
    {"id": 9, "pos": (72.500, 85.000, 17.500), "rot": (-90.00, 336.00, 0.00)},
    {"id": 10, "pos": (127.500, 0.000, 15.000), "rot": (-90.00, 270.00, 0.00)},
    {"id": 11, "pos": (132.500, -132.500, 15.000), "rot": (-90.00, 302.00, 0.00)},
    {"id": 12, "pos": (132.500, 127.500, 12.500), "rot": (-90.00, 56.00, 0.00)},
    {"id": 13, "pos": (-12.500, -90.000, 20.000), "rot": (-90.00, 90.00, 0.00)},
    {"id": 14, "pos": (-12.500, 2.500, 17.500), "rot": (-90.00, 0.00, 0.00)},
    {"id": 15, "pos": (-12.500, 87.500, 17.500), "rot": (-90.00, 0.00, 0.00)},
    {"id": 16, "pos": (-50.000, -42.500, 17.500), "rot": (-90.00, 0.00, 0.00)},
    {"id": 17, "pos": (-50.000, 42.500, 17.500), "rot": (-90.00, 0.00, 0.00)},
    {"id": 18, "pos": (-87.500, 0.000, 22.500), "rot": (-90.00, 0.00, 0.00)},
    {"id": 19, "pos": (-87.500, -92.500, 20.000), "rot": (-90.00, 0.00, 0.00)},
    {"id": 20, "pos": (-87.500, 87.500, 20.000), "rot": (-90.00, 0.00, 0.00)},
    {"id": 21, "pos": (-75.000, 145.000, 20.000), "rot": (-90.00, 72.00, 0.00)},
    {"id": 22, "pos": (-75.000, -137.500, 20.000), "rot": (-90.00, 296.00, 0.00)},
    {"id": 23, "pos": (-162.500, -92.500, 22.500), "rot": (-90.00, 0.00, 0.00)},
    {"id": 24, "pos": (-155.000, -5.000, 25.000), "rot": (-90.00, 0.00, 0.00)},
    {"id": 25, "pos": (-160.000, 87.500, 20.000), "rot": (-90.00, 0.00, 0.00)},
    {"id": 26, "pos": (-125.000, 45.000, 20.000), "rot": (-90.00, 0.00, 0.00)},
    {"id": 27, "pos": (-125.000, -45.000, 20.000), "rot": (-90.00, 0.00, 0.00)},
    {"id": 28, "pos": (-175.000, -137.500, 20.000), "rot": (-90.00, 54.00, 0.00)},
    {"id": 29, "pos": (-175.000, 145.000, 20.000), "rot": (-90.00, 296.00, 0.00)},

    # Overhead spectator
    {"id": "overhead", "pos": (-50, 0, 260), "rot": (-90, 0, 0)}
]

CAMERA_MAP = {cam["id"]: cam for cam in CAMERA_CONFIGS}

def synthesize_edge_events(edge_path):
    edge_events = []
    for file_path in glob(os.path.join(edge_path, "*.json")):
        base_name = os.path.basename(file_path)
        try:
            cam_id = int(base_name.split("_")[1])
        except (IndexError, ValueError):
            print(f"Skipping file with unexpected name format: {file_path}")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)
            for event in data:
                event["camera_id"] = cam_id
                event["time"] = event["frame"] / FPS
                edge_events.append(event)
    edge_events.sort(key=lambda e: e["frame"])
    return edge_events

def synthesize_inner_events(inner_path):
    inner_events = []
    for file_path in glob(os.path.join(inner_path, "*.json")):
        base_name = os.path.basename(file_path)
        try:
            cam_id = int(base_name.split("_")[1])
        except (IndexError, ValueError):
            print(f"Skipping file with unexpected name format: {file_path}")
            continue

        cam_info = CAMERA_MAP.get(cam_id)
        if cam_info is None:
            print(f"Warning: camera id {cam_id} not in CAMERA_CONFIGS. Skipping {file_path}")
            continue

        with open(file_path, "r") as f:
            data = json.load(f)
            for event in data:
                event["camera_id"] = cam_id
                event["location"] = cam_info["pos"]
                inner_events.append(event)
    inner_events.sort(key=lambda e: e["start_s"])
    return inner_events

def save_events(events, output_file):
    with open(output_file, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Saved {len(events)} events to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Synthesize edge and inner events into consolidated JSON files.")
    parser.add_argument("scenario", help="Top-level scenario folder containing edge_data and inner_data")
    args = parser.parse_args()

    scenario_path = args.scenario
    edge_path = os.path.join(scenario_path, "edge_data", "events")
    inner_path = os.path.join(scenario_path, "inner_data", "events")

    edge_events = synthesize_edge_events(edge_path)
    save_events(edge_events, os.path.join(scenario_path, "all_edge_events.json"))

    inner_events = synthesize_inner_events(inner_path)
    save_events(inner_events, os.path.join(scenario_path, "all_inner_events.json"))

if __name__ == "__main__":
    main()