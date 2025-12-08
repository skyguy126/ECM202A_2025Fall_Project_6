import numpy as np
from scapy.all import rdpcap
from scapy.layers.dot11 import Dot11
import os
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json

FPS = 20

# Kept at 2.0 as it successfully detected the car parts
SIGMA_SENSITIVITY = 2.0 

# Event must be at least 1.5 seconds to count
# Removes the 0.55s noise
MIN_DURATION_SECONDS = 1.5

# Wait 2.0 secs before deciding the car is gone
# Merges split events
GAP_TOLERANCE_SECONDS = 2.0

# Ignore the first 2 seconds to avoid startup noise
IGNORE_START_SECONDS = 2.0

def pcap_to_frame_sizes(pcap_path):
    print(f"Loading: {pcap_path}...")
    try:
        packets = rdpcap(pcap_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    start_time = 0
    first_idx = 0
    for idx, pkt in enumerate(packets):
        if pkt.haslayer(Dot11) and pkt[Dot11].type == 2 and len(pkt) > 1000:
            start_time = float(pkt.time)
            first_idx = idx
            break
            
    packets = packets[first_idx:]
    frame_dur = 1.0 / FPS
    sizes = []
    curr_frame_idx = 0
    curr_acc = 0
    
    for pkt in tqdm(packets, desc="Parsing Packets", leave=False):
        if pkt.haslayer(Dot11) and pkt[Dot11].type == 2:
            ts = float(pkt.time)
            f_idx = int((ts - start_time) / frame_dur)
            if f_idx > curr_frame_idx:
                for _ in range(f_idx - curr_frame_idx):
                    sizes.append(curr_acc)
                    curr_acc = 0
                curr_frame_idx = f_idx
                curr_acc += len(pkt)
            else:
                curr_acc += len(pkt)
    sizes.append(curr_acc)
    return np.array(sizes)

def analyze_motion(raw_sizes):
    # Remove i-frames
    iframe_threshold = np.percentile(raw_sizes, 95)
    clean_signal = raw_sizes.copy()
    clean_signal[clean_signal > iframe_threshold] = np.median(raw_sizes)
    
    # Smooth signal (p-frames)
    window = 20 
    smoothed = np.convolve(clean_signal, np.ones(window)/window, mode='same')
    
    # Calculate threshold based on current noise
    baseline = np.median(smoothed)
    std_dev = np.std(smoothed)
    
    min_buffer = 200 
    dynamic_buffer = std_dev * SIGMA_SENSITIVITY
    actual_buffer = max(dynamic_buffer, min_buffer)
    car_threshold = baseline + actual_buffer
    
    print(f"\nStats Analysis:")
    print(f" - Baseline Noise: {baseline:.0f} bytes")
    print(f" - Threshold Set:  > {car_threshold:.0f} bytes")

    # Find events
    events = []
    is_active = False
    start_frame = 0
    gap_counter = 0
    
    gap_limit_frames = int(GAP_TOLERANCE_SECONDS * FPS)
    min_frames = int(MIN_DURATION_SECONDS * FPS)
    
    # Calculate frames to ignore
    ignore_frames = int(IGNORE_START_SECONDS * FPS)
    
    for i, size in enumerate(smoothed):
        # ignore noise
        if i < ignore_frames:
            continue

        if size > car_threshold:
            if not is_active:
                is_active = True
                start_frame = i
            gap_counter = 0
        else:
            if is_active:
                gap_counter += 1
                # if signal jumps back then merge into one event
                if gap_counter >= gap_limit_frames:
                    end_frame = i - gap_limit_frames
                    if (end_frame - start_frame) >= min_frames:
                        events.append((start_frame, end_frame))
                    
                    is_active = False
                    gap_counter = 0

    if is_active:
        end_frame = len(smoothed)
        # check length of event
        if (end_frame - start_frame) >= min_frames:
             events.append((start_frame, end_frame))
                
    return events, clean_signal, smoothed, car_threshold

def save_events(events, clean_signal, smoothed, threshold, raw_sizes, out_dir, camera_id):

    events_file = f"camera_{camera_id}_events.json"
    events_dir = Path(os.path.join(out_dir, "events"))
    events_dir.mkdir(parents=True,exist_ok=True)
    events_path = Path(os.path.join(events_dir, events_file))

    plot_file = f"camera_{camera_id}_plot.png"
    plot_dir = Path(os.path.join(out_dir, "plots"))
    plot_dir.mkdir(parents=True,exist_ok=True)
    plot_path = os.path.join(plot_dir, plot_file)

    print("\n" + "="*45)
    print(f"DETECTED EVENTS: {len(events)}")
    print("="*45)
    print(f"{'#':<5} {'START (s)':<12} {'END (s)':<12} {'DURATION':<10}")
    print("-" * 45)

    # Prepare list of event dicts
    event_list = []
    for i, (start, end) in enumerate(events):
        s_time = start / FPS
        e_time = end / FPS
        dur = e_time - s_time
        
        start_fmt = f"{s_time:.2f}"
        if s_time > 60:
            m = int(s_time // 60)
            s = s_time % 60
            start_fmt += f" ({m}:{s:04.1f})"
            
        print(f"{i+1:<5} {start_fmt:<12} {e_time:<12.2f} {dur:<10.2f}s")

        # Add to JSON list
        event_list.append({
            "event": i+1,
            "start_s": s_time,
            "end_s": e_time,
            "duration": dur
        })

    print("="*45)

    # Write JSON file
    with open(events_path, "w") as f:
        json.dump(event_list, f, indent=2)

    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(raw_sizes, color='lightgray', label='Raw')
    plt.title("Raw Data")
    
    plt.subplot(2, 1, 2)
    plt.plot(clean_signal, color='lightblue', alpha=0.5, label='P-Frames')
    plt.plot(smoothed, color='darkblue', linewidth=2, label='Smoothed')
    plt.axhline(threshold, color='red', linestyle='--', label=f'Threshold')

    for (start, end) in events:
        plt.axvspan(start, end, color='green', alpha=0.3)
        
    plt.title(f"Motion Detection (Gap Tol: {GAP_TOLERANCE_SECONDS}s, Min Dur: {MIN_DURATION_SECONDS}s)")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(plot_path)
    print(f"\nVerification plot saved to: {plot_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert pcap to npy array of frame-level features"
    )
    parser.add_argument("pcap_file", help="Path to the pcap file to process. Enter a folder to process all pcaps in that folder.")
    parser.add_argument("-o", "--output_dir", type=str, help="Specify an output location. Default is the same directory as your input file.")

    args = parser.parse_args()

    pcap_path = Path(args.pcap_file)

    if pcap_path.is_dir():
        output_dir = pcap_path
        for pcap_file in pcap_path.glob("*.pcap"):
            camera_id = pcap_file.stem.split("_")[1].split(".")[0]

            raw_sizes = pcap_to_frame_sizes(str(pcap_file))
            events, clean_signal, smoothed, threshold = analyze_motion(raw_sizes)

            save_events(events, clean_signal, smoothed, threshold, raw_sizes, output_dir, camera_id)

    else:
        # just one pcap file to process 
        args.output_dir = str(Path(args.pcap_file).parent)
        camera_id = Path(args.pcap_file).stem.split("_")[1].split(".")[0]

        raw_sizes = pcap_to_frame_sizes(args.pcap_file)
        events, clean_signal, smoothed, threshold = analyze_motion(raw_sizes)

        save_events(events, clean_signal, smoothed, threshold, raw_sizes, args.output_dir, camera_id)

if __name__ == "__main__":
    main()