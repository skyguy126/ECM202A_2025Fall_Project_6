import carla
import util
import numpy as np
import cv2
import subprocess
import os
from queue import Queue, Empty

# Import camera configurations from util
CAMERA_CONFIGS = util.CAMERA_CONFIGS

def main():
    util.common_init()
    
    # Create videos directory if it doesn't exist
    videos_dir = "/media/ubuntu/Samsung/carla/demo/three_cars_1_6_8"
    os.makedirs(videos_dir, exist_ok=True)
    
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    util.check_sync(world)
    
    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(util.WIDTH))
    cam_bp.set_attribute("image_size_y", str(util.HEIGHT))
    cam_bp.set_attribute("fov", str(util.FOV))
    cam_bp.set_attribute("sensor_tick", str(1.0 / util.FPS))
    
    camera_data = []
    
    # Spawn all cameras and set up queues and ffmpeg processes
    for config in CAMERA_CONFIGS:
        camera_id = config["id"]
        pos = config["pos"]
        rot = config["rot"]
        
        cam_loc = carla.Location(x=pos[0], y=pos[1], z=pos[2])
        cam_rot = carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
        cam_tf = carla.Transform(cam_loc, cam_rot)
        
        camera = world.try_spawn_actor(cam_bp, cam_tf)
        if camera is None:
            print(f"Warning: Failed to spawn camera {camera_id} (position occupied). Skipping.")
            continue
        
        # Create queue for this camera and set up listener
        q = Queue()
        camera.listen(q.put)
        
        # Set up ffmpeg process for this camera
        filename = os.path.join(videos_dir, f"camera_{camera_id}.mp4")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",                    # overwrite output file
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",     # format we'll send from numpy
            "-s", f"{util.WIDTH}x{util.HEIGHT}",
            "-r", str(util.FPS),     # input frame rate (from util.py)
            "-i", "-",               # read video from stdin
            "-an",                   # no audio
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            filename,
        ]
        
        proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        camera_data.append({
            'camera': camera,
            'queue': q,
            'id': camera_id,
            'ffmpeg_proc': proc,
            'filename': filename
        })
        
        print(f"Camera {camera_id} recording to {filename}")
    
    print(f"\nSpawned {len(camera_data)} cameras. Recording started.")
    print("Press Ctrl+C or ESC to quit.\n")
    
    # Dictionary to store vehicle positions: {vehicle_id: [(frame_num, x, y), ...]}
    vehicle_positions = {}
    current_frame = 0  # Track current frame number (0-indexed)
    total_frames = 0  # Track total number of frames/ticks
    
    try:
        while True:
            # Advance the simulation by one fixed step
            world_frame = world.tick()
            total_frames += 1
            
            # Collect vehicle positions before processing frames (use 0-indexed frame number)
            util.collect_vehicle_positions(world, vehicle_positions, current_frame)
            current_frame += 1
            
            # Get frames from all cameras
            for cam_info in camera_data:
                try:
                    frame = cam_info['queue'].get(timeout=0.1)
                except Empty:
                    continue
                
                # Convert frame to numpy array (BGRA -> BGR)
                arr = np.frombuffer(frame.raw_data, np.uint8).reshape(
                    (frame.height, frame.width, 4)
                )[:, :, :3].copy()
                
                # Sanity: dimensions must match what we told ffmpeg
                assert frame.width == util.WIDTH and frame.height == util.HEIGHT
                
                # Write raw bytes to ffmpeg stdin
                try:
                    cam_info['ffmpeg_proc'].stdin.write(arr.tobytes())
                except BrokenPipeError:
                    print(f"Warning: ffmpeg process for camera {cam_info['id']} closed unexpectedly")
                
                # # Display frame in window named after camera ID
                # window_name = f"Camera {cam_info['id']}"
                # cv2.imshow(window_name, arr)
            
            # Process window events and check for ESC key
            # if cv2.waitKey(1) == 27:  # ESC key
            #    break
                
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down cameras and ffmpeg processes...")
        for cam_info in camera_data:
            cam_info['camera'].stop()
            cam_info['camera'].destroy()
            
            # Clean shutdown of ffmpeg
            if cam_info['ffmpeg_proc'].stdin:
                cam_info['ffmpeg_proc'].stdin.close()
            cam_info['ffmpeg_proc'].wait()
            print(f"Camera {cam_info['id']} saved to {cam_info['filename']}")
        
        # Save vehicle positions to numpy txt files
        print("\nSaving vehicle positions...")
        for vehicle_id, positions in vehicle_positions.items():
            if len(positions) > 0:
                # Extract frame numbers and positions
                # positions is a list of (frame_number, x, y) tuples
                first_frame = positions[0][0]  # Frame when vehicle first appeared
                last_frame = positions[-1][0]  # Frame when vehicle last appeared
                
                # Extract just the x, y coordinates
                xy_positions = np.array([[x, y] for _, x, y in positions])
                
                # Calculate padding needed
                pad_before = first_frame  # Zeros before vehicle appeared (frames 0 to first_frame-1)
                pad_after = total_frames - 1 - last_frame  # Zeros after vehicle disappeared
                
                # Create padding arrays
                padding_before = np.zeros((pad_before, 2)) if pad_before > 0 else np.empty((0, 2))
                padding_after = np.zeros((pad_after, 2)) if pad_after > 0 else np.empty((0, 2))
                
                # Concatenate: padding_before + positions + padding_after
                if pad_before > 0 and pad_after > 0:
                    positions_array = np.vstack([padding_before, xy_positions, padding_after])
                elif pad_before > 0:
                    positions_array = np.vstack([padding_before, xy_positions])
                elif pad_after > 0:
                    positions_array = np.vstack([xy_positions, padding_after])
                else:
                    positions_array = xy_positions
                
                # Ensure we have exactly total_frames rows
                assert len(positions_array) == total_frames, f"Expected {total_frames} frames, got {len(positions_array)}"
                
                # Save as txt file
                filename = os.path.join(videos_dir, f"vehicle_{vehicle_id}_positions.txt")
                np.savetxt(filename, positions_array, fmt='%.6f', delimiter=',')
                print(f"Vehicle {vehicle_id} positions saved to {filename} (padded to {len(positions_array)} frames: {pad_before} before, {len(xy_positions)} actual, {pad_after} after)")
        
        cv2.destroyAllWindows()
        print("All cameras stopped.")

if __name__ == "__main__":
    main()
