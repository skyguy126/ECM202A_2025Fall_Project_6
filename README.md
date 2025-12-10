# Urban Traﬃc Inference with Perimeter Vision & Encrypted-Camera Side-Channel Extraction

### Authors

- **Amy Lee** (siuyuetlee@ucla.edu, [GitHub](https://github.com/Harukk246))
- **Katherine Sohn** (katherinesohn@ucla.edu, [GitHub](https://github.com/katherinesohn)) 
- **Vamsi Eyunni** (veyunni@ucla.edu, [GitHub](https://github.com/skyguy126))

### [Project Website and Writeup](https://skyguy126.github.io/ECM202A_2025Fall_Project_6/)

---

## Software Setup

### Dependencies

The following dependencies are prerequsite to running any software for this project though the specific setup instructions are out of scope of this guide. Please refer to the below URLs for the most up-to-date setup guides from the respective project owners. As setup directions vary widely between different machines we recommend referring to the original dependency guides.

- Operating System: `Ubuntu 22.04.5 LTS`
- CUDA GPU with latest drivers. (i.e. `Driver Version: 550.107.02, CUDA Version: 12.4`)
- Python `3.7.17` running on the host machine. (Note: we suggest using [pyenv](https://github.com/pyenv/pyenv))
- [pylot](https://github.com/erdos-project/pylot)
    - This README assumes a functional deployment of the pylot docker image with full X11 forwarding and GPU passthrough to the host machine. Common caveats include pulling the `carla` Python library from the container's version of the [CARLA simulator](https://carla.org/) and installing it on the host. You may attempt to use `scritps/dev/create_dev_cont.sh` and `scripts/dev/run_cont.sh` but machine specific setup steps will vary.
- [Mininet Wi-Fi](https://mininet-wifi.github.io/)

### Start the CARLA Simulator

1. Create or connect to the `pylot` container:

```bash
# if you need to create a new container and make persistent changes within
./create_dev_cont.sh autocommit

# or if the container is already setup
./run_cont.sh
```

2. Start CARLA inside the docker container `./scripts/run_simulator.sh`

# WARNING
Do not run the above command (in step 1) on Vamsi's desktop. Instead use `./run_cont.sh`.

2. Setup and run the CARLA simulator:

```bash
cd scripts/dev
. ./start_carla.sh
```
This script also sets the python path (solves behaviorAgent not found error) and loads the town 5 map. 
Safe to re-run in order to reload town 5. 

2. Run a one-car scenario
```bash
cd scripts/cars
python one_car_route.py
```
Run with the `--help` flag for more options. 

OR, Run a multi-car scenario
```bash
cd scripts/cars
./multi_car_route.sh <route id 1> <route id 2>
```
Parameters are route ids, separated by spaces. Colors are randomly assigned. 

3. Run the cleanup script to stop CARLA: 
```bash
cd scripts/dev
./cleanup.sh
```
## Notes
- `tmux` may be useful for opening multiple windows in the docker container. 

# Camera Locations

![map](./docs/assets/img/map.png)

```
# Refer CAMERA_CONFIGS in util.py
```

## Start the CALRA simulator (on Vamsi's desktop)

```bash
cd ~/M202A-CARLA/scripts
./run_cont.sh
# At this point you should be on a shell within the docker container.
./scripts/run_simulator.sh
```

## Load World

```bash
# This command is run on the host.
python load_town5.py
```

## Start Cameras

```bash
# This command is run on the host.
python spawn_world5_cameras.py
```

This script will start the cameras at the hardocded locations above and start recording to mp4 (with ffmpeg). This script is responsible for advancing the world tick when Carla is running in sync mode.

The videos are output to `scripts/videos`.

### Warning

Do not advance the world tick with `world.tick()` in any other file.

## Sync Videos _to_ the Mininet VM

```bash
# This command is run on the host.
./scripts/mininet/push_video_to_mininet.sh
```

This will take the videos from `scripts/videos` and put it on `~/videos` on the Mininet VM.

## Run the wifi simulator and packet capture

```bash
# These commands are run inside the mininet vm.
cd ~/M202A-CARLA
sudo ./clean_mininet.sh
sudo python two_stations_wifi.py
```

The pcap files will be output to `~/M202A-CARLA/scripts/mininet/pcaps`.

## Copy PCAP files out of the Mininet VM

```bash
# This command is run on the host.
./scripts/mininet/sync_mininet_files.sh
```

The pcap files will be in `scripts/mininet/pcaps`.
