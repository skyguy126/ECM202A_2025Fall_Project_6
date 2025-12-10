# Urban Traﬃc Inference with Perimeter Vision & Encrypted-Camera Side-Channel Extraction

### Authors

- **Amy Lee** (siuyuetlee@ucla.edu, [GitHub](https://github.com/Harukk246))
- **Katherine Sohn** (katherinesohn@ucla.edu, [GitHub](https://github.com/katherinesohn)) 
- **Vamsi Eyunni** (veyunni@ucla.edu, [GitHub](https://github.com/skyguy126))

### **[Project Website and Writeup](https://skyguy126.github.io/ECM202A_2025Fall_Project_6/)**

---

## Software Setup

### Dependencies

The following dependencies are prerequsite to running any software for this project though the specific setup instructions are out of scope of this guide. Please refer to the below hyperlinks for the most up-to-date setup guides. As setup directions vary widely between different machines, we recommend referring to the project specific setup guides.

- Operating System: `Ubuntu 22.04.5 LTS`
- CUDA GPU with latest drivers. (i.e. `Driver Version: 550.107.02, CUDA Version: 12.4`)
- Python `3.7.17` running on the host machine. (Note: we suggest using [pyenv](https://github.com/pyenv/pyenv))
    - Install necessary dependencies with: `pip install -r ./requirements.txt`
- [pylot](https://github.com/erdos-project/pylot)
    - This README assumes a functional deployment of the pylot docker image with full X11 forwarding and GPU passthrough to the host machine. Common caveats include pulling the `carla` Python library from the container's version of the [CARLA simulator](https://carla.org/) and installing it on the host. You may attempt to use `scritps/dev/create_dev_cont.sh` and `scripts/dev/run_cont.sh` but machine specific setup steps will vary.
- [Mininet Wi-Fi](https://mininet-wifi.github.io/)
    - We recommend using the pre-made VirtualBox image and forward VM port 22 to host port 2222 for easy `ssh` access.
- ffmpeg: `sudo apt install ffmpeg`

### Start the CARLA Simulator

1. Create or connect to the `pylot` container:

```bash
# if you need to create a new container and make persistent changes within
./create_dev_cont.sh autocommit

# or if the container is already setup
./run_cont.sh
```

2. Start CARLA inside the docker container `./scripts/run_simulator.sh`
    - The aforementioned scripts also forward all ports to the host for easier development.
1. Load the map used in the writeup: `python ./scripts/dev/load_town5.py`
    - This will load the world used in the project writeup and move the observer camera to a birds eye view.

### Start Camera Capture

Now we will proceed to spawn cameras to emulate both edge and inner video feeds. Refer to the following diagram for hardcoded camera positions used in the writeup.

![map](./docs/assets/img/map.png)

To enable or disable cameras, comment and uncomment lines in `./scripts/util.py` within the `CAMERA_CONFIGS` array. By default all cameras are enabled.

1. Modify the input and output paths defined in `./scripts/spawn_world5_cameras.py`. The MP4 recordings will be saved accordingly.
1. Start camera capture: `python ./scripts/spawn_world5_cameras.py`.
    - Note that this will transition the CARLA simulator to sync mode until the simulator is restarted.
1. To end camera capture press `Ctrl-C`.

### Spawn Vehicles

In this section you will spawn one or more vehicles with either custom or predefined routes. The predefined routes are provided for convinience and quick reproducability.

1. `./scripts/multi_car_route <route ID #1> <route ID #2> <route ID #N>`
    - Route IDs are defined in [this spreadsheet](https://docs.google.com/spreadsheets/d/1RwLaQ0M5xuA0Guw3VsfJDIJVlorXgBcg69KmpaDnyVU/edit?gid=65310523#gid=65310523).

The script will spawn and start vehicle movement, and will automatically exit once all vehicles have reached their destination waypoint.

### Pre-Process Edge Camera Video

At this point this guide will assume you have a set of MP4 files with recordings of car(s) driving through the city in CARLA. *Camera 4* and *Camera 5* are currently hardcoded to act as "edge cameras". Therefore we will process these two MP4 files as edge camera video to obtain a dataset containing various parameters such as assigning vehicles a global identifcation number through the multi-vantage tracking system.

1. Modify hardcoded paths in `./scripts/process_edge_camera_video.py` as needed and launch the processing script with `python ./scripts/process_edge_camera_video.py`.
     - This script will output `.json` files in the output directory you specified.

### Generate Inner Camera Network Data (MP4 -> PCAP)

1. Start the Mininet Wi-Fi VM.
1. Clone this repository to `/home/wifi`.
1. Copy over inner camera MP4s from the host to the mininet VM. Tools such as `scp` may come in handly.
     - Remmeber that only `camera_4.mp4`, `camera_5.mp4`, and `camera_overhead.mp4` are edge/debug videos. All other camera IDs are inner camera video files. These are the files you will need to copy to the mininet VM.
1. Within the mininet VM, modify the paths as needed; defined at the top of `./scripts/mininet/two_stations_wifi.py`.
1. Run the mininet virtual wifi setup: `sudo python ./scripts/mininet/two_stations_wifi.py`.
    - The script will initialize the network architecture then enter a CLI. Type `exit` to allow the script to continue processing video files.
    - The script will then automatically loop through each inner camera video file and stream it between stations. The sniffer will capture network packets and save them to the `./scripts/mininet/pcaps` folder.
1. Copy the `./scripts/mininet/pcaps` folder from the mininet VM to the host for further analysis.

### Build Dataset

Once you have generated edge camera event data (`.json` files) and inner camera network data (`.pcap` files), combine them into a consolidated dataset:

1. Ensure you have the following files prepared:
   - Edge camera `.json` files in `<scenario>/edge_data/events/`
   - Inner camera `.pcap` files processed and their event `.json` files in `<scenario>/inner_data/events/`

2. Synthesize all events into final consolidate event files. Choose one of the following approaches:
   
   **Option A: Manual synthesis**
   - Run the final synthesis script: `python ./scripts/data_parsing/get_final_events.py <scenario_folder>`
   - Replace `<scenario_folder>` with the path to your scenario directory (e.g., `./demos/one_car_2/`)
   - This script will:
     - Parse all edge camera event `.json` files and synthesize them into `all_edge_events.json`
     - Parse all inner camera event `.json` files with camera position metadata and synthesize them into `all_inner_events.json`
     - Output both consolidated files to the scenario root directory

   **Option B: Automated workflow**
   - Run the shell script: `./scripts/data_parsing/parse_scenario.sh <scenario_name>`
   - This will automatically handle all parsing steps including edge events, inner events, and final synthesis

3. The resulting `all_edge_events.json` and `all_inner_events.json` files located at the base level of the scenario folder `demos/<scenario_name>` can now be used for inference and analysis.

### Perform Inference

With your consolidated event data ready, perform inference to reconstruct vehicle trajectories:

1. Run the inference algorithm. Choose one of the following approaches:

   **Option A: Kalman Filter Algorithm**
   - Run: `python ./M202A_algorithm2.py <scenario_name>`
   - Replace `<scenario_folder>` with your scenario name (e.g., `one_car_2`)
   - This will output `final_trajectory.csv` in the scenario directory

   **Option B: Graph Optimization Algorithm**
   - Run: `python ./graph_algorithm.py <scenario_name>`
   - Replace `<scenario_folder>` with your scenario name (e.g., `one_car_2`)
   - This will output `graph_trajectory.csv` in the scenario directory

### Generate Visuals

Visualize the results with the plot script:
   - For KF algorithm results: `python ./plot_final_trajectory.py <scenario_name>`
   - For GO algorithm results: `python ./plot_final_trajectory.py <scenario_name> -g`
   - The `-g` flag indicates to plot the graph algorithm output
   - Visualization plots will be saved to the scenario directory as or `trajectory_plot.png` or `graph_alg_plot.png`, respectively

To re-run all base demos through both algorithms and plot all results, run the `./refresh_plots.sh` script. This script is currently ad-hoc and not parameterized, but can be easily refactored to re-run results for any demo algorithm. 

### Run Error Analysis 

Run error analysis for all scenarios in the specified folder (default `demos/`): `python ./comparison.py <folder>`
---

## Experimental Machine Learning Models

These models and scripts are highly experimental and are provided in hopes of providing inspiration for future work rather than optimizing for ease of use. Please contact project authors for exact details but the rough steps are outlined below.

### Machine Learning based PCAP Feature Extraction

This model is architected to:

1. Accept input of all inner camera `.pcap` files and pre-process them into `X` and `y` training data.
1. Train a LSTM-based model to learn hidden features in the PCAP feature vectors.
1. Perform inference on new `.pcap` files to determine timing of vehicle events.


### Execution Instructions

1. First generate ground truth data with `python ./scripts/mininet/parse_video.py`.
    - Modify paths at the top of the file as needed. The video folder path must point to the location where *all*, both edge and inner, videos are stored.
1. Generate feature vectors based on the collection of `.pcap` files from mininet. Run `./scripts/mininet/parse_pcap.py`.
    - Adjust the paths as needed. `PCAPS_DIR` should point to the collecton of `.pcap` files obtained from: *Generate Inner Camera Network Data (MP4 -> PCAP)*.
1. Train the machine learning mdoe with `python ./scripts/mininet/model.py`.
    - Adjust the paths at the top of the file as needed.
    - The general inputs to this file are:
        - `X` = feature vector(s) processed from the raw `.pcap` files.
        - `Y` = ground truth classification provided by `./scripts/mininet/parse_viceo.py`.
1. Perform inference by specifying the necessary agruments to `python ./scripts/mininet/infer.py.`
 