import carla
import time
import random
import sys
import signal
from agents.navigation.basic_agent import BasicAgent
from util import FPS
import numpy as np

random.seed(42)
np.random.seed(42)

TM_PORT = 8000  # port for traffic manager
TOWN_NAME = "Town05"

# add debugging for print statements
import builtins, traceback
_real_print = builtins.print
def hook(*a, **k):
    s = " ".join(map(str, a))
    if "deque index out of range" in s:
        traceback.print_stack(limit=12)
    _real_print(*a, **k)
builtins.print = hook

def handle_sigterm(signum, frame):
    raise KeyboardInterrupt  # convert SIGTERM into KeyboardInterrupt

# Register the handler
signal.signal(signal.SIGTERM, handle_sigterm)

def load_town(client):
    print("loading town...")

    world = client.load_world(TOWN_NAME)

    spectator = world.get_spectator()
    # High altitude straight-down
    location = carla.Location(x=-50, y=0, z=260)
    rotation = carla.Rotation(pitch=-90, yaw=0, roll=0)
    spectator.set_transform(carla.Transform(location, rotation))

    print("Spectator moved to bird's eye position.")
    print("loaded town")

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
    route_points = [random.choice(inside_spawns) for _ in range(1000)]

    print("Spawn at:", route_points[0].location)

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
    # vehicle.set_autopilot(False, TM_PORT)  # important! BehaviorAgent controls it manually
    vehicle.set_autopilot(False)

    # -----------------------------
    # Initialize the agent
    # -----------------------------
    # agent = BehaviorAgent(vehicle, ignore_traffic_light=True, behavior="normal")
    agent = BasicAgent(vehicle, target_speed=30)

    print("Starting route...")

    # Tick the world a few times so everything initializes
    for _ in range(5):
        world.tick()

    # -------------------------------------------
    # Simulation loop: destination-only, distance-based heuristic per waypoint
    # -------------------------------------------
    try:
        print_interval = 20  # print every 20 ticks (~1 second if tick = 0.05s)

        for t in route_points[1:]:

            # Reject waypoint if behind the car or would cause route planner to U-turn
            loc = vehicle.get_location()
            transform = vehicle.get_transform()
            forward = transform.get_forward_vector()
            dx = t.location.x - loc.x
            dy = t.location.y - loc.y
            dist_to_wp = (dx * dx + dy * dy) ** 0.5
            if dist_to_wp >= 1e-3:
                dir_x = dx / dist_to_wp
                dir_y = dy / dist_to_wp
                dot = forward.x * dir_x + forward.y * dir_y
                # dot < 0: behind; dot < 0.5: sharp turn / potential U-turn (~60° off heading)
                if dot < 0.77:
                    print("Skipping waypoint (behind vehicle or would cause U-turn):", t.location)
                    continue
            else:
                print("Skipping waypoint (already at waypoint):", t.location)
                continue

            print("changing to new destination:", t.location)
            # Changed: BasicAgent just needs the target location
            # Changed: Older BasicAgent requires a tuple (x, y, z), not a Location object
            agent.set_destination((t.location.x, t.location.y, t.location.z))

            for _ in range(5):
                world.tick()

            tick_counter = 0

            while True:

                # Changed: BasicAgent handles its own queue state elegantly
                if agent.done():
                    break

                control = agent.run_step()
                vehicle.apply_control(control)
                world.tick()

                dist = vehicle.get_location().distance(t.location)
                if tick_counter % print_interval == 0:
                    print(f"Distance to waypoint: {dist:.2f} meters")
                tick_counter += 1

                if dist < 20:  # distance tolerance to move to next waypoint
                    print("breaking due to distance heruistic")
                    break

        print("Reached destination.")
        
    finally:
        print("Destroying vehicle...")
        vehicle.destroy()

        # tick a few times to destroy the vehicle properly
        for _ in range(5):
            world.tick()

        # TODO: SANITY CHECK NUM FRAMES == TIMESTEPS
        # TODO:
        # TODO:

def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)

    load_town(client)
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
