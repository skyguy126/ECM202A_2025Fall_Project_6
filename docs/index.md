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

Modern cities rely on surveillance for traffic management, but continuous tracking of vehicles through urban zones remains challenging due to privacy and coverage gaps. We address this problem by fusing perimeter vision (entry/exit cameras) with interior encrypted-video side-channels (wireless traffic data) to infer vehicle trajectories without direct observation. Our approach integrates computer vision for vehicle detection at the zone perimeter with analysis of encrypted camera WiFi traffic to signal motion inside. We develop a novel, multi-target tracking system to maintain estimated vehicle position with spatial and temporal stability. In evaluation using the CARLA simulator, the system tracks multiple cars through a virtual city, matching ground truth trajectories within acceptable error. These results demonstrate the feasibility of “seeing without seeing", achieving situational awareness via indirect sensing, and highlight the broader potential of side-channel data to augment urban traffic monitoring while mitigating privacy concerns.

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

Emerging smart cities grapple with the trade-off between ubiquitous surveillance for traffic management and privacy intrusion. Our project explores a novel approach to urban vehicle tracking that leverages video coverage at a zone’s perimeter and side-channel analysis of encrypted camera feeds within the zone. By combining these heterogeneous data sources: high-fidelity visual detection at entry/exit points and coarse motion cues from encrypted network traffic, we aim to continuously track vehicles as they roam within an urban area.

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
- Edge cameras were placed overhead to allow for clean data collection but this is often not represtntative of reality.
- The simulated vehicles move at a constant velocity and do not stop per usual traffic rules. This is not reflective of reality as most vehicles operate with perodic acceleration and deceleration.

### **3.1 System Architecture**
Include a block diagram or pipeline figure.

### **3.2 Data Pipeline**
Explain how data is collected, processed, and used.

#### CARLA Setup

The CARLA simulator is instianted via the `pylot` docker container and all ports are exposed on the host. The simulator is run within a docker container to ensure reproducability and easy sharing of dependencies. All ephermal code and data collection is executed on the host itself.

#### Camera Capture

Static RGB cameras are placed at fixed, repeatable poses so every trial observes identical viewpoints; controlled camera geometry improves cross-run comparability and is standard practice in multi-camera tracking benchmarks. Each camera samples at 20 FPS with 1280×720 resolution and a 90° field of view to balance spatial detail with real-time throughput (similar rates are used in KITTI/nuScenes to match perception pipelines). Frames arrive as raw bytes and are first buffered in per-camera queues to decouple acquisition from storage, a common technique in real-time vision systems to prevent frame drops when I/O stalls.

Frames are compressed on the fly with an intra/long-GOP H.264 encoder fed via stdin. Piping raw frames directly into the encoder avoids intermediate disk writes and aligns with recommendations from the video systems literature for reducing latency and preserving quality in real-time capture (e.g., FFmpeg-based pipelines in robotics and teleoperation studies). Queue draining is synchronized with the simulator tick, and watchdog checks flag size mismatches or encoder failures so other cameras continue uninterrupted.

Vehicle poses are logged once per tick in world coordinates (x, y; ground plane), assuming a flat road surface. Pre- and post-padding with zeros yields a fixed-length trajectory per vehicle, which simplifies later alignment between tracks and videos; fixed-length temporal representations are widely used in trajectory forecasting datasets to enable direct frame-indexed fusion.

#### Mininet WiFi

Camera-to-access-point links are emulated in software with Mininet-WiFi to retain the full Linux TCP/IP stack while avoiding specialized radio hardware. A minimal topology with one access point and two stations mirrors common sender/receiver lab setups and keeps contention controlled. The emulator’s `wmediumd` interference model injects realistic wireless effects (loss, rate adaptation) instead of idealized zero-loss pipes, following recommendations from the Mininet-WiFi authors for fidelity in SDN and wireless research.

Over-the-air traffic is captured via a monitor-mode interface and recorded per video as PCAPs; this mirrors methodology in traffic-analysis and side-channel studies where timing, burstiness, and packet sizes—not payloads—are the primary signals. Transmissions are replayed at source frame rate using application-layer streaming so temporal characteristics match the encoded video. Each capture is closed after a clip to ensure one-to-one alignment between videos and PCAPs, improving downstream feature extraction reliability.

Mininet-WiFi is chosen because it offers repeatable experiments, real kernel code paths, and scriptable control at low cost, as documented in the SDN and wireless emulation literature (e.g., Mininet-WiFi design and evaluation papers). It also enables rapid iteration compared with physical testbeds while providing more realism than packet-level simulators.

### **3.3 Algorithm / Model Details**

#### **Edge Camera** Multi-Vantage Tracking

We run two prerecorded edge videos (cameras 4 and 5) frame-by-frame. Each frame goes through YOLOv8x for detection plus ByteTrack for short-term tracking, which yields a box and a per-camera track id for each vehicle. We crop the box and pass it to an OSNet ReID network to get a 512-D appearance embedding; OSNet is used because it is lightweight and pretrained for person/vehicle re-identification, so it works well without heavy fine-tuning. The bottom-center of each box is projected into world coordinates using calibrated intrinsics/extrinsics so both cameras report positions in the same frame. A global appearance tracker keeps a cross-camera gallery: cosine similarity (threshold 0.65) links new embeddings to existing global IDs, otherwise it spawns a new one. To smooth noise, gallery embeddings are updated with a running average (80% previous, 20% new). We log per-frame JSON with camera pose, global/local IDs, and estimated world positions; this compact log is later used by the fusion step without needing to replay video. Key design choices for non-experts: YOLO+ByteTrack gives robust boxes and stable short tracks; appearance-only matching (no timing/GPS) avoids needing synchronization; the similarity threshold trades off false merges vs. splits; the running average keeps IDs stable even if a single frame is noisy.

#### **Inner-Camera** Side Channel PCAP Feature Extraction

##### Deterministic Approach

WIP Katherine

##### Machine Learning Approach

**Methodology**

Packet traces are converted into per-frame descriptors by isolating 802.11 data traffic after the first high-volume video packet and aggregating over the camera frame period (1/FPS). Each frame records packet count, total bytes, mean and variance of packet sizes, inter-arrival statistics, and index bounds; empty frames are zero-filled to preserve alignment. Paired video-derived features are matched by camera identifier, truncated to equal length, and standardized after discarding the first 500 frames to suppress startup transients—steps consistent with traffic-analysis practice. Three packet-side channels (count, total bytes, size variability) serve as predictors, while video embeddings supply supervision.

Overlapping windows of 16 frames are generated with stride 1 to retain fine temporal structure while expanding the effective dataset. An 80/20 split yields train/test partitions. A two-layer bidirectional LSTM (hidden size 128, dropout 0.1) maps each window to per-timestep predictions, optimized with mean squared error and Adam (learning rate 3e-4) for 75 epochs. This bi-directional, windowed formulation mirrors standard sequence-to-sequence regression setups, enabling the model to leverage both past and future context within each clip.

**Results**

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
