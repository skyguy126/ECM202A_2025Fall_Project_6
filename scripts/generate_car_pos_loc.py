import carla
import time
import random
import sys
import signal
from behavior_agent import BehaviorAgent
from util import FPS
import numpy as np

random.seed(42)
np.random.seed(42)

# -------------------------------------------
# Configuration (edit these instead of using CLI args)
# -------------------------------------------
TM_PORT = 8000  # port for traffic manager

def handle_sigterm(signum, frame):
    raise KeyboardInterrupt  # convert SIGTERM into KeyboardInterrupt

# Register the handler
signal.signal(signal.SIGTERM, handle_sigterm)

def worker(client, world):

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    
    w_map = world.get_map()

    # -------------------------------------------
    # Set synchronous mode
    # -------------------------------------------
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enable sync mode
    settings.fixed_delta_seconds = 1.0 / FPS 
    world.apply_settings(settings)

    # -----------------------------
    # Generate route
    # -----------------------------
    spawns = w_map.get_spawn_points()
    inside_spawns = []
    outside_spawns = []
    for sp in spawns:
        if -300 <= sp.location.x <= 180 and -180 <= sp.location.y <= 180:
            inside_spawns.append(sp)
        else:
            outside_spawns.append(sp)

    # We can change the number of waypoints internally.
    route_points = [
        random.choice(inside_spawns),
        random.choice(inside_spawns),
        random.choice(inside_spawns),
        random.choice(inside_spawns),
    ]

    print("Spawn at:", route_points[0].location)

    # Convert transforms (after spawnpoint) to waypoints
    route_waypoints = [] 
    for t in route_points[1:]:
        wp = w_map.get_waypoint(t.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        route_waypoints.append(wp)
        print(f"Waypoint: x={wp.transform.location.x:.2f}, y={wp.transform.location.y:.2f}, z={wp.transform.location.z:.2f}")

    # -------------------------------------------
    # Spawn the vehicle
    # -------------------------------------------
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.toyota.prius")

    print("Spawning vehicle...")
    vehicle = world.try_spawn_actor(vehicle_bp, route_points[0])

    if vehicle is None:
        print("Failed to spawn vehicle.")
        return

    world.player = vehicle 
    vehicle.set_autopilot(False, TM_PORT)  # important! BehaviorAgent controls it manually

    # -----------------------------
    # Initialize the agent
    # -----------------------------
    agent = BehaviorAgent(vehicle, ignore_traffic_light=False, behavior="normal")
   
    print("Starting route...")

    # Tick the world a few times so everything initializes
    for _ in range(5):
        world.tick()

    # -------------------------------------------
    # Simulation loop
    # -------------------------------------------
    try:
        first = True
        for wp in route_waypoints:
            wp_loc = wp.transform.location
            print(f"Next waypoint: x={wp_loc.x:.2f}, y={wp_loc.y:.2f}, z={wp_loc.z:.2f}")

            # Set the current waypoint as the destination
            agent.set_destination(vehicle.get_location(), wp_loc, clean=first)
            first = False

            tick_counter = 0
            print_interval = 20  # print every 20 ticks (~1 second if tick = 0.05s)

            # Loop until we reach this waypoint
            while True:
                try:
                    agent.update_information(world)
                    control = agent.run_step()
                    vehicle.apply_control(control)
                    world.tick()

                    # Compute distance to waypoint
                    dist = vehicle.get_location().distance(wp_loc)

                    # # Only print every print_interval ticks
                    if tick_counter % print_interval == 0:
                        print(f"Distance to waypoint: {dist:.2f} meters")

                    tick_counter += 1

                    # this never gets hit because of the exception
                    # Check if we are close enough to the current waypoint
                    if dist < 4.0:  # 2-meter tolerance
                        break
                except Exception as e:
                    print(e, "agent error, next waypoint")
                    break

        print("Reached destination.")
        
    finally:
        print("Destroying vehicle...")
        vehicle.destroy()
        # tick a few times to destroy the vehicle properly
        for _ in range(5):
            world.tick()

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # start worker function
    worker(client, world)

    # reset to async mode so we don't freeze the simulator
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(0)
