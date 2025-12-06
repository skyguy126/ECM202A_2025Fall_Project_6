import carla
import time

# ----------------------------
# Configuration
# ----------------------------
VEHICLE_BLUEPRINT = 'vehicle.toyota.prius'  # change to any vehicle you want
SPAWN_FILE = 'spawn_points.txt'

# ----------------------------
# Helper functions
# ----------------------------
def load_spawn_points(file_path):
    """Load spawn points from a simple txt file"""
    spawn_points = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, z, pitch, yaw, roll = map(float, line.strip().split())
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)
            )
            spawn_points.append(transform)
    return spawn_points

# ----------------------------
# Connect to CARLA
# ----------------------------
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find(VEHICLE_BLUEPRINT)

# Load spawn points
spawn_points = load_spawn_points(SPAWN_FILE)
print(f"Loaded {len(spawn_points)} spawn points")

# ----------------------------
# Interactive loop
# ----------------------------
vehicle = None  # current spawned vehicle

try:
    while True:
        index = input(f"\nEnter spawn point index (0-{len(spawn_points)-1}, or 'q' to quit): ")
        if index.lower() == 'q':
            break
        
        try:
            index = int(index)
            if index < 0 or index >= len(spawn_points):
                print("Invalid index")
                continue
        except ValueError:
            print("Please enter a number")
            continue

        # Spawn vehicle
        transform = spawn_points[index]
        vehicle = world.spawn_actor(vehicle_bp, transform)
        
        loc = vehicle.get_transform().location
        print(f"Spawned vehicle at: {loc.x} {loc.y} {loc.z}")

        # Wait for Enter to despawn
        input("Press Enter to despawn vehicle...")
        vehicle.destroy()
        vehicle = None
        print("Vehicle despawned")

finally:
    if vehicle is not None:
        vehicle.destroy()
    print("Exiting script")
