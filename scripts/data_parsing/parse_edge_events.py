import json
import argparse
import os
from pathlib import Path

OUTPUT_DIR = "out/"

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

def write_event_data_to_file(event_data, input_file, out_dir=OUTPUT_DIR):
    """Write event data to a JSON file in the same dir as input file."""
    # Create out/ directory if it doesn't exist
    out_dir = Path(input_file).parent
    out_dir = Path(os.path.join(out_dir, "events"))
    out_dir.mkdir(exist_ok=True)
    
    # Get the input filename without extension
    input_path = Path(input_file)
    output_filename = Path(input_file).stem.replace("_input", "") + "_events.json"
    output_path = out_dir / output_filename
    
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
    with open(output_path, "w") as f:
        json.dump(events_list, f, indent=2)
    
    print(f"Events written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse car detection events from JSON.")
    parser.add_argument("file", help="Path to camera input JSON file")
    args = parser.parse_args()

    data = load_data(args.file)
    events = get_detection_events(data)
    print_events(events)
    
    event_data = extract_event_data(events, data)
    print_event_data(event_data)
    
    write_event_data_to_file(event_data, args.file)

if __name__ == "__main__":
    main()