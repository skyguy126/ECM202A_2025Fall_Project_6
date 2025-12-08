import carla
import time
import random
import sys
import argparse 
import os
import signal
from behavior_agent import BehaviorAgent
from util import FPS

def handle_sigterm(signum, frame):
    raise KeyboardInterrupt  # convert SIGTERM into KeyboardInterrupt

# Register the handler
signal.signal(signal.SIGTERM, handle_sigterm)

COLOR_MAP = {
    "red":      "255,0,0",
    "green":    "0,255,0",
    "blue":     "0,0,255",
    "yellow":   "255,255,0",
    "cyan":     "0,255,255",
    "magenta":  "255,0,255",
    "white":    "255,255,255",
    "black":    "0,0,0",
}
VALID_COLOR_NAMES = ", ".join(COLOR_MAP.keys())

def main():

    # -----------------------------
    # Parse command-line arguments
    # -----------------------------
    parser = argparse.ArgumentParser(description="Scenario Parameters")
    parser.add_argument(
        "-w", "--write",
        action="store_true",
        default=False,
        help="write the route to a file"
    )
    parser.add_argument(
        "-r", "--read",
        action="store_true",
        default=False,
        help="read the route from ID"
    )
    parser.add_argument(
        "-i", "--id",
        type=int,
        required=False,
        default=1,
        help="ID of the route to read/write"
    )
    parser.add_argument(
        "-t", "--traffic-lights",
        action="store_true",
        default=False,
        help="Respect traffic lights"
    )
    parser.add_argument(
        "--tm-port",
        type=int,
        default=8000,
        help="port to communicate with the traffic manager"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="1",
        help="name of car in the simulation"
    )
    parser.add_argument(
        "--color",
        type=str,
        default="red",
        help=f"Color of the vehicle in the simulation. Valid names: {VALID_COLOR_NAMES}"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        default=False,
        help="run standalone without cameras"
    )
    args = parser.parse_args()

    # -------------------------------------------
    # Connect to CARLA
    # -------------------------------------------
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    world = client.get_world()
    w_map = world.get_map()

    # -------------------------------------------
    # Set synchronous mode
    # -------------------------------------------
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enable sync mode
    settings.fixed_delta_seconds = 1.0 / FPS 
    world.apply_settings(settings)

    # -----------------------------
    # Load route from file
    # -----------------------------
    route_points = []
    if not args.read:
        # three point route generation 
        spawns = w_map.get_spawn_points()
        # sort spawns into inside/outside highway
        inside_spawns = []
        outside_spawns = []
        for sp in spawns:
            if -300 <= sp.location.x <= 180 and -180 <= sp.location.y <= 180:
                inside_spawns.append(sp)
            else:
                outside_spawns.append(sp)
        # Pick points: spawn/outside, middle/inside, destination/outside
        route_points.append(random.choice(outside_spawns))
        route_points.append(random.choice(inside_spawns))
        route_points.append(random.choice(outside_spawns))

        if args.write:
            # if file exists, warn user
            if os.path.exists(f"routes/{args.id}.txt"):
                input(f"WARNING: You're about to overwrite route {args.id}. Press Enter to continue...")
            with open(f"routes/{args.id}.txt", "w") as f:
                for t in route_points:
                    loc = t.location
                    rot = t.rotation
                    f.write(f"{loc.x} {loc.y} {loc.z} {rot.pitch} {rot.yaw} {rot.roll}\n")
    else:
        with open(f"routes/{args.id}.txt", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 6:
                    continue
                x, y, z, pitch, yaw, roll = map(float, parts)
                transform = carla.Transform(
                    carla.Location(x=x, y=y, z=z),
                    carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
                )
                route_points.append(transform)


    # print("spawn at:", route_points[0].location)

    # Convert transforms (after spawnpoint) to waypoints
    route_waypoints = [] 
    for t in route_points[1:]:
        wp = w_map.get_waypoint(t.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        route_waypoints.append(wp)
        # print(f"Waypoint: x={wp.transform.location.x:.2f}, y={wp.transform.location.y:.2f}, z={wp.transform.location.z:.2f}")

    # -------------------------------------------
    # Spawn the vehicle
    # -------------------------------------------
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find("vehicle.toyota.prius")
    if args.color:
        color_input = args.color.lower()
        color_value = COLOR_MAP.get(color_input, args.color)
    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", color_value)

    print(f"Spawning vehicle {args.name}...")
    vehicle = world.try_spawn_actor(vehicle_bp, route_points[0])

    if vehicle is None:
        print(f"Failed to spawn vehicle {args.name}.")
        return

    world.player = vehicle 

    vehicle.set_autopilot(False, args.tm_port)  # important! BehaviorAgent controls it manually

    # -----------------------------
    # Initialize the agent
    # -----------------------------
    ignore_traffic_light = not args.traffic_lights
    agent = BehaviorAgent(vehicle, ignore_traffic_light=ignore_traffic_light, behavior="normal")
   
    print(f"Starting route for car {args.name}...")

    # Tick the world a few times so everything initializes
    for _ in range(5):
        world.tick()

    # -------------------------------------------
    # Simulation loop
    # -------------------------------------------
    try:
        for wp in route_waypoints:
            wp_loc = wp.transform.location
            print(f"Next waypoint for car {args.name}: x={wp_loc.x:.2f}, y={wp_loc.y:.2f}, z={wp_loc.z:.2f}")

            # Set the current waypoint as the destination
            agent.set_destination(vehicle.get_location(), wp_loc, clean=True)

            tick_counter = 0
            print_interval = 20  # print every 20 ticks (~1 second if tick = 0.05s)

            # Loop until we reach this waypoint
            while True:
                try: 
                    if(args.run):
                        world.tick()
                    else:
                        world.wait_for_tick()
                    agent.update_information(world)
                    control = agent.run_step()
                    vehicle.apply_control(control)

                    # Compute distance to waypoint
                    dist = vehicle.get_location().distance(wp_loc)

                    # # Only print every print_interval ticks
                    # if tick_counter % print_interval == 0:
                    #     print(f"Distance to waypoint: {dist:.2f} meters")

                    tick_counter += 1

                    # this never gets hit because of the exception
                    # Check if we are close enough to the current waypoint
                    if dist < 2.0:  # 2-meter tolerance
                        break
                except Exception as e: 
                    print(e)
                    print("agent error, next waypoint")
                    break

        print(f"{args.name} reached destination.")
        
    finally:
        print(f"Destroying {args.name}...")
        vehicle.destroy()
        # tick a few times to destroy the vehicle properly
        for _ in range(5):
            world.tick()

    # settings.synchronous_mode = False
    # settings.fixed_delta_seconds = None
    # world.apply_settings(settings)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(0)
