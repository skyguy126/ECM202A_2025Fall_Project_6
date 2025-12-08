---
layout: default
title: "Urban Traﬃc Inference with Perimeter Vision & Encrypted-Camera Side-Channel Extraction"
---

# **Urban Traﬃc Inference with Perimeter Vision & Encrypted-Camera Side-Channel Extraction**

*A concise, descriptive title for your project.*

![Project Banner](./assets/img/banner-placeholder.png)  
<sub>*(Optional: Replace with a conceptual figure or meaningful image.)*</sub>

---

## 👥 **Team**

- **Amy Lee** (siuyuetlee[at]ucla[dot]edu, [GitHub](https://github.com/Harukk246))
- **Katherine Sohn** (katherinesohn[at]ucla[dot]edu, [GitHub](https://github.com/katherinesohn)) 
- **Vamsi Eyunni** (veyunni[at]ucla[dot]edu, [GitHub](https://github.com/skyguy126))

---

## 👥 **Mentors**

- **Gaofeng Dong** (gfdong[at]ucla[dot]edu)
- **Pragya Sharma** (pragyasharma[at]ucla[dot]edu)

---

## 📝 **Abstract**

Provide a short paragraph (4–6 sentences) summarizing:

- The problem you addressed  
- Your approach  
- Key results  
- The significance of your findings  

This should be a standalone “TL;DR” of your project.

---

## 📑 **Slides**

- [Midterm Checkpoint Slides](https://docs.google.com/presentation/d/1X-M2zomr7aMnQWdz8OJdX9qPV1sSsXszmZ0EHS2_K9M/edit?usp=sharing)  
- [Final Presentation Slides](https://docs.google.com/presentation/d/14e4P2keh30SPoivBp-G8eDYCar3hYOaEy6AgSDjk5yw/edit?usp=sharing)
- If Needed: Links to have any videos demonstrating your project

---

> Delete this section if you have no videos.

## 🎛️ **Media**

- Links to any video or audio recordings demonstrating your project

---

# **1. Introduction**

Use the introduction to clearly set context, describe motivation, and explain the central idea behind your project.

### **1.1 Motivation & Objective**  
What are you trying to do and why is it important? Explain in accessible, non-jargon terms.

### **1.2 State of the Art & Its Limitations**  
How is this problem addressed today?  
What gaps or limitations exist?  
Cite prior work using a consistent style like [Smith21].

### **1.3 Novelty & Rationale**  
What is new about your approach, and why do you expect it to succeed?

### **1.4 Potential Impact**  
If successful, what technical or broader impact could this project have?

### **1.5 Challenges**  
List the main technical, practical, or methodological challenges.

### **1.6 Metrics of Success**  
What are the specific, measurable criteria for evaluating your project?

---

# **2. Related Work**

Summarize prior works relevant to your project.  
For each: what did the authors do, how is it related, and what gap remains?

Reference all citations in **Section 6**.

---

# **3. Technical Approach**

Describe your system, methodology, algorithms, and design choices.  
Use figures generously:

- System architecture diagram  
- Data pipeline  
- Algorithm/model block diagram  
- Hardware setup photos  

💡 Tip: Add images, diagrams, and code snippets. Make your system reproducible.

Recommended subsections:

### **3.0 Assumptions & Proposed Solutions**

- At any given time, only one car occupies any camera's field of view.
- Each camera transmits only to its associated access point, with no other traffic on that link; therefore all WiFi traffic observed on that network originates from the camera.
- The WiFi sniffer is positioned so it is within range of only a single camera/access-point pair at any given time, preventing capture of unrelated over-the-air traffic.
- The multi-vantage identification system generates embeddings solely from vehicle appearance without incorporating temporal or spatial metadata. It performs reliably when vehicle colors differ significantly, but error rates may increase when vehicles have similar appearances.
- The CARLA simulator runs in synchronous mode. This can reduce control accuracy because the control loop operates at a lower tick rate (20 FPS to match the camera frame rate). Asynchronous simulation would normally decouple the control algorithm from perception processing and maintain a fixed timestep, but due to resource constraints this project uses CARLA in synchronous mode.
- The edge camera multi-vantage tracking approah uses a hardcoded value for the cosine similarity threshold. This value should be dynamically derived based on current scene and environment conditions to ensure stability between different CARLA maps and vehicles.
- This project uses deterministic vehicle route mapping to further allow control for result replication.

### **3.1 System Architecture**
Include a block diagram or pipeline figure.

### **3.2 Data Pipeline**
Explain how data is collected, processed, and used.

#### CARLA Setup

The CARLA simulator is instianted via the `pylot` docker container and all ports are exposed on the host. The simulator is run within a docker container to ensure reproducability and easy sharing of dependencies. All ephermal code and data collection is executed on the host itself.

#### Camera Capture

We place multiple static RGB cameras in the map using the preset poses in `util.CAMERA_CONFIGS`, so every run sees the same streets from the same angles. Each camera is 1280×720, 90° FOV, and runs at 20 FPS to match the simulator tick. CARLA delivers frames as raw bytes; we push them into a per-camera queue to decouple capture from disk writes (avoids dropped frames if storage hiccups).

For storage we stream raw BGR frames to `ffmpeg` over stdin and encode with H.264 into per-camera MP4 files (`/media/ubuntu/Samsung/carla/demo/two_cars_6_8/camera_<id>.mp4`). This piped approach keeps quality high while using a widely supported codec. The control loop calls `world.tick()`, drains queues, sanity-checks frame size, and logs warnings if an encoder fails—other cameras keep running so a single failure doesn’t stop the experiment.

At the same tick rate we log every vehicle’s world (x, y) position. The z position is assumed to be 0. We pad zeros before a vehicle appears and after it leaves so each `vehicle_<id>_positions.txt` has exactly one row per frame; this makes it easy to align trajectories with the videos later. (References: piping raw frames to FFmpeg for Python workflows is a standard pattern; see ffmpeg-python docs and common Stack Overflow examples.)

#### Mininet WiFi

We emulate camera-to-AP WiFi links in software using Mininet-WiFi to avoid needing physical radios while still running the real Linux TCP/IP stack. The `two_stations_wifi.py` topology keeps things minimal: one access point (`ap1`) and two stations (`sta1` sender, `sta2` receiver). We use `wmediumd` with an interference model so packets still traverse a simulated wireless channel rather than a zero-loss virtual wire.

Before streaming, we bring up a monitor-mode interface (`hwsim0`) and start `tcpdump` to capture over-the-air traffic into per-video PCAPs. `sta1` replays each MP4 with `ffmpeg -re ... -f mpegts udp://10.0.0.201:5000`, while `sta2` listens with `ffmpeg` and discards payloads; only timing, lengths, and bursts matter for side-channel features. After each video, captures are closed cleanly to keep PCAPs aligned to a single clip.

Why Mininet(-WiFi)? It runs real kernel networking on one machine, scales with modest resources, and exposes a Python API to script topologies, mobility, and SDN-style control. That makes it faster to iterate than hardware testbeds while remaining more realistic than pure packet-level simulators; the community and docs (e.g., mininet.org) also simplify debugging and reproducibility.
### **3.3 Algorithm / Model Details**

#### **Edge Camera** Multi-Vantage Tracking

We run two prerecorded edge videos (cameras 4 and 5) frame-by-frame. Each frame goes through YOLOv8x for detection plus ByteTrack for short-term tracking, which yields a box and a per-camera track id for each vehicle. We crop the box and pass it to an OSNet ReID network to get a 512-D appearance embedding; OSNet is used because it is lightweight and pretrained for person/vehicle re-identification, so it works well without heavy fine-tuning. The bottom-center of each box is projected into world coordinates using calibrated intrinsics/extrinsics so both cameras report positions in the same frame. A global appearance tracker keeps a cross-camera gallery: cosine similarity (threshold 0.65) links new embeddings to existing global IDs, otherwise it spawns a new one. To smooth noise, gallery embeddings are updated with a running average (80% previous, 20% new). We log per-frame JSON with camera pose, global/local IDs, and estimated world positions; this compact log is later used by the fusion step without needing to replay video. Key design choices for non-experts: YOLO+ByteTrack gives robust boxes and stable short tracks; appearance-only matching (no timing/GPS) avoids needing synchronization; the similarity threshold trades off false merges vs. splits; the running average keeps IDs stable even if a single frame is noisy.

#### **Inner-Camera** Side Channel PCAP Feature Extraction

##### Deterministic Approach

WIP Katherine

##### Machine Learning Approach

WIP Vamsi

#### Final Fusion Algorithm

WIP Katherine/Amy 

### **3.4 Hardware / Software Implementation**
Explain equipment, libraries, or frameworks.

### **3.5 Key Design Decisions & Rationale**
Describe the main design decisions you made.

---

# **4. Evaluation & Results**

Present experimental results with clarity and professionalism.

Include:

- Plots (accuracy, latency, energy, error curves)  
- Tables (comparisons with baselines)  
- Qualitative visualizations (spectrograms, heatmaps, bounding boxes, screenshots)  
- Ablation studies  
- Error analysis / failure cases

Each figure should have a caption and a short interpretation.

---

# **5. Discussion & Conclusions**

Synthesize the main insights from your work.

- What worked well and why?  
- What didn’t work and why?  
- What limitations remain?  
- What would you explore next if you had more time?  

This should synthesize—not merely repeat—your results.

---

# **6. References**

Provide full citations for all sources (academic papers, websites, etc.) referenced and all software and datasets uses.

---

# **7. Supplementary Material**

## **7.a. Datasets**

Describe each dataset:
* Source and URL
* Data format
* Preprocessing steps
* Labeling/annotation efforts

Include your internal dataset if you collected one.
## **7.b. Software**

List:
* External libraries or models
* Internal modules you wrote
* Links to repos or documentation

---

> [!NOTE] 
> Read and then delete the material from this line onwards.

# 🧭 **Guidelines for a Strong Project Website**

- Include multiple clear, labeled figures in every major section.  
- Keep the writing accessible; explain acronyms and algorithms.  
- Use structured subsections for clarity.  
- Link to code or datasets whenever possible.  
- Ensure reproducibility by describing parameters, versions, and preprocessing.  
- Maintain visual consistency across the site.

---

# 📊 **Minimum vs. Excellent Rubric**

| **Component**        | **Minimum (B/C-level)**                                         | **Excellent (A-level)**                                                                 |
|----------------------|---------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Introduction**     | Vague motivation; little structure                             | Clear motivation; structured subsections; strong narrative                                |
| **Related Work**     | 1–2 citations; shallow summary                                 | 5–12 citations; synthesized comparison; clear gap identification                          |
| **Technical Approach** | Text-only; unclear pipeline                                  | Architecture diagram, visuals, pseudocode, design rationale                               |
| **Evaluation**       | Small or unclear results; few figures                          | Multiple well-labeled plots, baselines, ablations, and analysis                           |
| **Discussion**       | Repeats results; little insight                                | Insightful synthesis; limitations; future directions                                      |
| **Figures**          | Few or low-quality visuals                                     | High-quality diagrams, plots, qualitative examples, consistent style                      |
| **Website Presentation** | Minimal formatting; rough writing                           | Clean layout, good formatting, polished writing, hyperlinks, readable organization        |
| **Reproducibility**  | Missing dataset/software details                               | Clear dataset description, preprocessing, parameters, software environment, instructions   |
