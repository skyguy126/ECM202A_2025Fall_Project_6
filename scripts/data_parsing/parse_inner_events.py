"""
Thresholding-based car detection from encrypted 802.11 frames.

This script analyzes packet patterns in pcap files to detect when cars pass
in front of cameras. The approach leverages the fact that:
- ffmpeg encodes video in I, P, and B frames
- I-frames (keyframes) are larger and occur when scene changes
- When a car enters the frame, there's more motion, leading to more/larger packets per frame
- Static backgrounds produce relatively constant packet patterns


By grouping packets into frames, then thresholding frame sizes, we can detect
when activity increases, indicating a car is in the camera's field of view.
"""
import numpy as np
from scapy.all import rdpcap
from scapy.layers.dot11 import Dot11
import os
from pathlib import Path
import sys
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

FPS = 20

# ----------------------------
# CONFIG
# ----------------------------

OUTPUT_DIR = "tmp/"
MIN_VIDEO_PACKET_SIZE = 1000  # Minimum packet size to consider as video transmission packet

# ----------------------------
# Frame-level feature extraction function
# ----------------------------

def pcap_to_frame(pcap_path):
    """
    convert pcap file of packets to video frames 
    """
    # Load pcap file
    packets = rdpcap(pcap_path)
    
    if len(packets) == 0:
        raise RuntimeError(f"No packets found in {pcap_path}")
    
    # Find the first video transmission packet (802.11 data frame with larger size)
    first_video_packet = None
    first_video_timestamp = None
    first_video_packet_index = None
    
    for idx, pkt in enumerate(packets):
        # Check if it's an 802.11 data frame (type == 2)
        if pkt.haslayer(Dot11):
            dot11 = pkt[Dot11]
            # Type 2 = Data frame
            if dot11.type == 2:
                packet_size = len(pkt)
                if packet_size >= MIN_VIDEO_PACKET_SIZE:
                    first_video_packet = pkt
                    first_video_timestamp = float(pkt.time)
                    first_video_packet_index = idx
                    break
    
    if first_video_packet is None:
        raise RuntimeError(f"No video transmission packet found in {pcap_path}")
       
    # chop off until first video packet
    packets = packets[first_video_packet_index:]

    # Calculate frame time window
    frame_duration = 1.0 / FPS  # seconds per frame
    
    # Collect all 802.11 data frames with their timestamps and sizes
    data_frames = []
    frame_idx = 0
    packet_acc = 0
    for idx, pkt in enumerate(tqdm(packets, desc="Collecting 802.11 data frames", leave=False)):
        # Check if it's an 802.11 data frame
        if pkt.haslayer(Dot11):
            dot11 = pkt[Dot11]
            if dot11.type == 2:  # Data frame
                timestamp = float(pkt.time)
                # if this is the next frame
                if (timestamp - first_video_timestamp)/frame_duration >= frame_idx:
                    data_frames.append([frame_idx, packet_acc])
                    frame_idx += 1
                    packet_acc = 0
                # else accumulate the packet size to the current frame
                else:
                    packet_acc += len(pkt)
    
    if len(data_frames) == 0:
        raise RuntimeError(f"No 802.11 data frames found after first video packet in {pcap_path}")
    
    print(f"Found a total of {len(data_frames)} frames")    
    print("Duration in sec:", len(data_frames)/FPS)

    return np.array(data_frames)

def frames_to_events(data, args):
    """Convert frame-level data to car detection events."""
    events = []
    final_events = []

    normalized_data = data[:,1].copy()

    mean = np.mean(normalized_data)
    std = np.std(normalized_data)
    normalized_data = (normalized_data - mean) / std

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # All Data
    axs[0].plot(data[:,0], data[:,1])
    axs[0].set_xlabel("Frame Index")
    axs[0].set_ylabel("Frame Size")
    axs[0].set_title("Frame Size vs Frame Index")

    # i-frames removed
    axs[1].plot(data[:,0], normalized_data)  
    axs[1].set_xlabel("Frame Index")
    axs[1].set_ylabel("Frame Size")
    axs[1].set_title("Frame Size vs Frame Index (Normalized)")

    plt.tight_layout()
    plt.show()

    # don't include noise in mean
    mean = normalized_data[normalized_data > 0.01].mean()
    print(np.sum(normalized_data[normalized_data > 0.01]), " frames are bigger than 0.01")

    win_length = args.window
    threshold_size = mean
    required_fraction = args.threshold

    print("More than ", required_fraction * win_length, " out of ", win_length, " must exceed ", threshold_size)
    events.clear()  # clear previous events
    final_events.clear()
    for idx, size in enumerate(normalized_data[0:-win_length]):
        over_count = np.sum(normalized_data[idx:idx+win_length] > threshold_size)
        if over_count / float(win_length) >= required_fraction:
            events.append(idx)

    if len(events) > 0:
        # prune events down to single events
        final_events.append(events[0])
        for idx in range(1,len(events)):
            if events[idx] - events[idx-1] > 30 and idx > 10: # ignore beginning noise
                final_events.append(events[idx])

    print(f"Detected events at indices: {final_events}")
    
    return final_events

def print_events(events):
    """Print detected events."""
    for event in events:
        print(f"detected an event at frame {event}")

def plot_frame_data(data, output_png="tmp/frame_plot.png"):

    # print(f"Data shape: {data.shape}")
    # print(f"Data dtype: {data.dtype}")
    # print(f"First few entries:\n{data[:5]}")
    
    # Extract frame numbers and packet sizes
    if data.dtype == object:
        # Data is structured array with named fields
        frames = np.array([entry['frame'] for entry in data])
        sizes = np.array([entry['size'] for entry in data])
    else:
        # Try to extract assuming first column is frame, second is size
        frames = data[:, 0]
        sizes = data[:, 1]
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(frames, sizes, 'b-', linewidth=1, alpha=0.7)
    plt.scatter(frames, sizes, s=20, alpha=0.5, color='blue')
    
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Packet Size (bytes)', fontsize=12)
    plt.title('Frame Number vs Packet Size', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(str(output_png), dpi=100, bbox_inches='tight')
    print(f"\nPlot saved to: {output_png}")
    
    # Display statistics
    print(f"  Total frames: {len(frames)}")
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Convert pcap to npy array of frame-level features"
    )
    parser.add_argument("-f", "--pcap-file", type=str, help="Path to the pcap file to process")
    parser.add_argument("-t", "--threshold", type=float, default=0.15, help="Replace default threshold of 15%% of window")
    parser.add_argument("-W", "--window", type=int, default=100, help="Replace default window width of 100")

    args = parser.parse_args()

    data = pcap_to_frame(args.pcap_file)
    plot_frame_data(data)
    events = frames_to_events(data, args)
    print_events(events)

if __name__ == "__main__":
    main()

