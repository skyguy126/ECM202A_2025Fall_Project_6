import json
import argparse
import os
from pathlib import Path

def load_data(path):
    with open(path, "r") as f:
        return json.load(f)

def get_detection_events(data):
    events = []
    active_event = None

    for entry in data:
        frame = entry["frame"]
        detected = entry["car_detected"]

        if detected:
            if active_event is None:
                # Start new event - extract global_id from first car
                global_id = None
                if entry["cars"]:
                    global_id = entry["cars"][0]["global_id"]
                active_event = {
                    "start_frame": frame,
                    "end_frame": frame,
                    "global_id": global_id
                }
            else:
                # Extend event
                active_event["end_frame"] = frame
        else:
            if active_event is not None:
                # Close event
                events.append(active_event)
                active_event = None

    # Close final event if open
    if active_event is not None:
        events.append(active_event)

    # Merge events from same car if less than 100 frames apart
    merged_events = []
    for event in events:
        if merged_events and merged_events[-1]["global_id"] == event["global_id"]:
            # Check if gap is less than 100 frames
            gap = event["start_frame"] - merged_events[-1]["end_frame"] - 1
            if gap < 100:
                # Merge events
                merged_events[-1]["end_frame"] = event["end_frame"]
                continue
        merged_events.append(event)

    return merged_events

def extract_event_data(events, data):
    """Convert events to list of (frame, car_id, location) tuples using middle frame."""
    event_data = []
    
    # Create a dictionary for quick lookup: frame -> entry
    # because we might chop off the beginning, so we can't just index data by frame
    frame_lookup = {entry["frame"]: entry for entry in data}
    
    for event in events:
        # Calculate middle frame
        current_frame = (event["start_frame"] + event["end_frame"]) // 2
        
        # Search forward by increments of 5 until we find car_detected=true
        while current_frame <= event["end_frame"]:
            if current_frame in frame_lookup:
                frame_entry = frame_lookup[current_frame]
                if frame_entry["car_detected"] and frame_entry["cars"]:
                    # Get position from first car (should match the event's car_id)
                    position = frame_entry["cars"][0]["position"]
                    event_data.append((current_frame, event["global_id"], position))
                    break
            current_frame += 5
    
    return event_data

def print_events(events):
    print("\n=== Car Detection Events ===")
    for i, e in enumerate(events):
        print(f"Event {i+1}: Car ID {e['global_id']} | frames {e['start_frame']} → {e['end_frame']} "
              f"(duration {e['end_frame'] - e['start_frame'] + 1} frames)")
    print("============================\n")

def print_event_data(event_data):
    print("\n=== Event Data (frame, car_id, location) ===")
    for frame, car_id, location in event_data:
        print(f"Frame {frame}: Car ID {car_id} | Position {location}")
    print("==========================================\n")

def write_event_data_to_file(event_data, input_file, out_dir, camera_id):

    events_file = f"camera_{camera_id}_events.json"
    events_dir = Path(os.path.join(out_dir, "events"))
    events_dir.mkdir(parents=True,exist_ok=True)
    events_path = Path(os.path.join(events_dir, events_file))

    # Convert event_data tuples to dictionaries for JSON serialization
    events_list = [
        {
            "frame": frame,
            "car_id": car_id,
            "location": location
        }
        for frame, car_id, location in event_data
    ]
    
    # Write to JSON file
    with open(events_path, "w") as f:
        json.dump(events_list, f, indent=2)
    
    print(f"Events written to {events_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse car detection events from JSON.")
    parser.add_argument("file", help="Path to camera input JSON file. Enter a directory to parse all json files in that directory.")
    parser.add_argument("-o", "--output_dir", type=str, help="Specify an output location. Default is events/ in the same directory as your input file.")

    args = parser.parse_args()

    json_path = Path(args.file)

    if json_path.is_dir():
        output_dir = json_path
        for json_file in json_path.glob("*.json"):
            camera_id = json_file.stem.split("_")[1]

            print(f"Parsing {json_file}...")
            data = load_data(json_file)
            events = get_detection_events(data)
            
            event_data = extract_event_data(events, data)
            print_event_data(event_data)
            
            write_event_data_to_file(event_data, json_file, output_dir, camera_id)

    else:
        output_dir = str(Path(args.pcap_file).parent)
        camera_id = json_file.stem.split("_")[1]
        
        data = load_data(args.file)
        events = get_detection_events(data)
        print_events(events)
        
        event_data = extract_event_data(events, data)
        print_event_data(event_data)
        
        write_event_data_to_file(event_data, args.file, output_dir, camera_id)

if __name__ == "__main__":
    main()