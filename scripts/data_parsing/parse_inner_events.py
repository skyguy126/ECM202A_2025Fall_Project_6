import numpy as np
from scapy.all import rdpcap
from scapy.layers.dot11 import Dot11
import os
import sys
import json
from tqdm import tqdm
from pathlib import Path
import argparse
from scipy.signal import medfilt

# --- GLOBAL CONFIGURATION ---
FPS = 20  

# --- DEFAULT SETTINGS ---
STD_MIN_DURATION_S = 2.0    # Events shorter than this are treated as noise/glitches
GAP_TOLERANCE_S = 1.0       # How long to wait for signal to return before ending an event
IGNORE_START_S = 10.0       # Ignore the beginning 10 seconds
MIN_BYTES_FLOOR = 100       # Minimum byte count to consider a signal 

# --- CAMERA SPECIFIC OVERRIDES ---
CAMERA_OVERRIDES = {
    # CASE A: High Noise (Cam 18)
    "18": {
        "TRIGGER_BUFFER": 0.6,   # Multiplier for Std Dev to set trigger threshold
        "MIN_DURATION_S": 5.0,   # Longer duration
        "FORCE_WINDOW": 20,      # Smoothing window size
        "SUSTAIN_RATIO": 0.3,    # How far the signal can drop before event ends
        "CLIP_SIGMA": 2.5        # Outlier rejection strictness
    },
    
    # CASE B: Cam 19
    "19": {
        "TRIGGER_BUFFER": 2.0,   
        "MIN_DURATION_S": 4.0,   
        "FORCE_WINDOW": 20,      
        "SUSTAIN_RATIO": 0.4,    
        "CLIP_SIGMA": 3.0        
    },

    # CASE C: Cam 20
    "20": {
        "TRIGGER_BUFFER": 1.0,   
        "MIN_DURATION_S": 4.0,   
        "FORCE_WINDOW": 20,      
        "SUSTAIN_RATIO": 0.2,    
        "CLIP_SIGMA": 2.0        
    },

    # CASE D: Cam 9
    "9": {
        "TRIGGER_BUFFER": 1.8,   
        "MIN_DURATION_S": 4.0,   
        "FORCE_WINDOW": 20,      
        "SUSTAIN_RATIO": 0.2,    
        "CLIP_SIGMA": 2.0        
    }
}

def pcap_to_frame_sizes(pcap_path):
    print(f"Loading: {pcap_path}...")
    try:
        packets = rdpcap(str(pcap_path))
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

def calibrate_noise_profile(signal, sigma_clip=3.0):
    """
    Calculates the background noise statistics (Mean, Std, Max).
    Iteratively removes outliers so they don't skew the noise calculation.
    """
    noise_samples = signal.copy()
    
    # Run 3 passes of clipping to remove high-value spikes 
    for _ in range(3):
        median = np.median(noise_samples)
        std = np.std(noise_samples)
        cutoff = median + (sigma_clip * std) 
        noise_samples = noise_samples[noise_samples < cutoff]
        
    # Safety check if signal was empty
    if len(noise_samples) == 0: 
        return np.median(signal), 100, 100 
        
    return np.mean(noise_samples), np.std(noise_samples), np.max(noise_samples)

def analyze_motion(raw_sizes, use_high_sensitivity=False, override_config=None):
    """
    Core detection algorithm. 
    1. Filters signal.
    2. Calculates dynamic thresholds based on noise.
    3. Iterates through signal to find events.
    """
    if len(raw_sizes) == 0:
        return [], 0, 0 

    # Median Filter to remove single-frame spikes (glitches)
    clean_signal = medfilt(raw_sizes, kernel_size=5)

    settings = {
        "CLIP_SIGMA": 3.0,
        "TRIGGER_BUFFER": 2.0,       # High threshold = Noise Max + (2.0 * Std)
        "SUSTAIN_RATIO": 0.4,        # Low threshold location (percentage between mean and high thresh)
        "FORCE_WINDOW": None,
        "MIN_DURATION_S": STD_MIN_DURATION_S
    }

    # High sensitivity for fallback
    if use_high_sensitivity:
        settings.update({
            "CLIP_SIGMA": 2.0,       # Clip tighter to find smaller noise floor
            "TRIGGER_BUFFER": 1.0,   # Lower trigger threshold
            "SUSTAIN_RATIO": 0.2,
            "FORCE_WINDOW": 20,      
            "MIN_DURATION_S": 4.0
        })

    # If a specific Camera Override exists, it overwrites everything else
    if override_config:
        settings.update(override_config)

    # Unpack settings for easier access
    CLIP_SIGMA = settings["CLIP_SIGMA"]
    TRIGGER_BUFFER = settings["TRIGGER_BUFFER"]
    SUSTAIN_RATIO = settings["SUSTAIN_RATIO"]
    FORCE_WINDOW = settings["FORCE_WINDOW"]
    MIN_DURATION_S = settings["MIN_DURATION_S"]

    # Determine Smoothing Window
    start_idx = int(IGNORE_START_S * FPS)
    calib_slice = clean_signal[start_idx:] if len(clean_signal) > start_idx else clean_signal
    
    noise_mean, noise_std, noise_max = calibrate_noise_profile(calib_slice, sigma_clip=CLIP_SIGMA)
    
    if FORCE_WINDOW:
        smooth_window = FORCE_WINDOW
    else:
        # Coefficient of Variation determines how messy the signal is.
        cv = noise_std / (noise_mean + 1e-5) 
        if cv > 0.5: smooth_window = 40 
        elif cv > 0.2: smooth_window = 20 
        else: smooth_window = 10          

    # Apply smoothing convolution
    smoothed = np.convolve(clean_signal, np.ones(smooth_window)/smooth_window, mode='same')
    
    # Recalibrate noise on the SMOOTHED signal for accurate thresholds
    calib_smooth = smoothed[start_idx:] if len(smoothed) > start_idx else smoothed
    s_mean, s_std, s_max = calibrate_noise_profile(calib_smooth, sigma_clip=CLIP_SIGMA)
    
    # Calculate Dual Thresholds (Schmitt Trigger)
    thresh_high = max(s_max + (TRIGGER_BUFFER * s_std), MIN_BYTES_FLOOR)
    
    # thresh_low is calculated as a percentage distance between the mean and the high threshold
    thresh_low = s_mean + (thresh_high - s_mean) * SUSTAIN_RATIO
    thresh_low = max(thresh_low, MIN_BYTES_FLOOR)

    # Event Detection Loop
    valid_events = []
    
    is_active = False
    start_frame = 0
    gap_counter = 0
    
    gap_limit_frames = int(GAP_TOLERANCE_S * FPS)
    min_frames = int(MIN_DURATION_S * FPS)
    
    for i, val in enumerate(smoothed):
        if i < start_idx: continue

        if not is_active:
            # Signal exceeds High Threshold
            if val > thresh_high:
                is_active = True
                start_frame = i
                gap_counter = 0
        else:
            # Signal stays above Low Threshold
            if val > thresh_low:
                gap_counter = 0
            else:
                gap_counter += 1
                
                # If gap persists too long, the event ends
                if gap_counter >= gap_limit_frames:
                    end_frame = i - gap_limit_frames
                    duration_frames = end_frame - start_frame
                    
                    # Average bytes during event must allow threshold
                    event_avg = np.mean(smoothed[start_frame:end_frame])
                    is_valid_quality = True
                    if (use_high_sensitivity or override_config) and event_avg < thresh_low:
                        is_valid_quality = False

                    # Must be longer than minimum duration
                    if duration_frames >= min_frames and is_valid_quality:
                        valid_events.append((start_frame, end_frame))
                    
                    is_active = False
                    gap_counter = 0
            
    if is_active:
        end_frame = len(smoothed)
        duration_frames = end_frame - start_frame
        
        event_avg = np.mean(smoothed[start_frame:end_frame])
        is_valid_quality = True
        if (use_high_sensitivity or override_config) and event_avg < thresh_low:
            is_valid_quality = False
            
        if duration_frames >= min_frames and is_valid_quality:
             valid_events.append((start_frame, end_frame))
                
    return valid_events, noise_mean, noise_std

def save_events_to_json(events, out_dir, camera_id):
    """
    Saves detected events to JSON only.
    """
    events_file = f"camera_{camera_id}_events.json"
    events_dir = Path(os.path.join(out_dir, "events"))
    events_dir.mkdir(parents=True, exist_ok=True)
    events_path = Path(os.path.join(events_dir, events_file))

    event_list = []
    for i, (start, end) in enumerate(events):
        s_time = (start / FPS) + 1.0 # +1.0 offset
        e_time = (end / FPS) + 1.0
        dur = e_time - s_time
        
        event_list.append({
            "event": i+1, 
            "start": round(s_time, 2), 
            "end": round(e_time, 2), 
            "duration": round(dur, 2)
        })

    with open(events_path, "w") as f:
        json.dump(event_list, f, indent=2)
    
    print(f"Camera {camera_id}: Saved {len(events)} events to {events_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pcap_file", help="Path to pcap file or folder")
    parser.add_argument("-o", "--output_dir", type=str)
    args = parser.parse_args()
    
    path = Path(args.pcap_file)
    files = list(path.glob("*.pcap")) if path.is_dir() else [path]
    out_dir = Path(args.output_dir) if args.output_dir else (path if path.is_dir() else path.parent)

    for f in files:
        try: cam_id = f.stem.split("_")[1].split(".")[0]
        except: cam_id = f.stem
        
        raw_sizes = pcap_to_frame_sizes(str(f))
        
        if cam_id in CAMERA_OVERRIDES:
            print(f"Processing Camera {cam_id} (Override Mode)...")
            config = CAMERA_OVERRIDES[cam_id]
            events, _, _ = analyze_motion(raw_sizes, override_config=config)
            
        else:
            print(f"Processing Camera {cam_id} (Standard Mode)...")
            events, n_mean, n_std = analyze_motion(raw_sizes, use_high_sensitivity=False)
            
            should_use_high_sens = False
            
            # No events found -> Try High Sensitivity
            if len(events) == 0:
                should_use_high_sens = True
                
            # High Noise Floor -> Try High Sensitivity
            elif n_mean > 2000 and n_std > 500:
                should_use_high_sens = True

            if should_use_high_sens:
                events_r, _, _ = analyze_motion(raw_sizes, use_high_sensitivity=True)
                
                # Only keep high sensitivity results if it actually found something
                if len(events_r) > 0:
                    events = events_r
                elif len(events) == 0:
                     pass

        save_events_to_json(events, out_dir, cam_id)

if __name__ == "__main__":
    main()