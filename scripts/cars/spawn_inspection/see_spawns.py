import carla
import random
import time

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(5.0)
world = client.get_world()

# Optional: set synchronous mode
settings = world.get_settings()
settings.synchronous_mode = False
world.apply_settings(settings)

# Get all spawn points
spawn_points = world.get_map().get_spawn_points()

# Get a vehicle blueprint
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find("vehicle.tesla.model3")  # pick any vehicle

# Keep track of spawned vehicles
vehicles = []

# Spawn a vehicle at each spawn point
for i, sp in enumerate(spawn_points):
    vehicle = world.try_spawn_actor(vehicle_bp, sp)
    if vehicle:
        vehicles.append(vehicle)
        # print(f"Spawned vehicle {i} at {sp.location}")
    else:
        print(f"Could not spawn vehicle at {sp.location}")

print(f"Spawned {len(vehicles)} vehicles in total.")

try:
    print("Press Ctrl+C to destroy vehicles and exit...")
    while True:
        time.sleep(1)
finally:
    print("Destroying vehicles...")
    for v in vehicles:
        v.destroy()
    print("Done.")