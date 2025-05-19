---
title: Engineering Portfolio

---
<a href="#top"> </a>

| Project Name           | Key Areas                              |
|------------------------|----------------------------------------|
| [PRM-Based Global Body Planner for Quadruped Robot](#planner) |  Planning |
| [Robust Control for Low-Mass Quadrotors under Wind Disturbances](#drone) | Controls, Embedded Systems |
| [RL for Autonomous Humanoid Bi-Manipulation](#boardwalk) | Reinforcement Learning, internship  |
| [Sketch-to-Image Diffusion Model](#imagediff) | Deep Learning |
| [Model-based Reinforcement Learning and Transformer Architecture in a Humanoid Robot Environment](#humrl) | Reinforcement Learning |
| [Hand Slapper: A Reaction Time Game](#slapper) | Embedded Systems, Circuit Design  |
| [Grasp Optimization from Learning-Based Initial Guess](#thesis) | Research, Controls, Reinforcement Learning |
| [Sound Localization and Autonomous Navigation](#slam) | SLAM, Planning |
| [Tour into the picture (implementation)](#cv) | Computer Vision  |
| [Self-Balancing trajectory following Robot](#balancing) | Controls, Embedded Systems |
| [Design of a Controller for a Buck-Boost Converter](#buck) | Circuit Design, Controls, Embedded Systems |
| [Internship at Siemens Healthineers](#siemens) | Internship, General Hardware, Embedded Systems |
| [Offline A-star Planner for Catching a Moving Target in an Arbitrary Map](#astar) | Planning |
| [Sampling-Based Planners for multi DoF Robotic Arm](#PRM) | Planning  |
| [CMA-ES and Imitation Learning for Bipedal Walker Control](#biped) | Reinforcement Learning |

# Real-Time Indoor Mapping with RTAB-Map: CPU-Level Parallelism and Descriptor Selection for Precise 3D Reconstruction
<a name="rtabmap"></a>
**High-Level Overview**  
This project extends RTAB-Map’s real-time SLAM pipeline by integrating CPU-level parallelism and evaluating feature pipelines, resulting in up to **1.8×** mapping throughput without sacrificing map precision using a rigorous ground-truth evaluation framework :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

**Background**  
Simultaneous Localization and Mapping (SLAM) requires a balance between computational efficiency and mapping accuracy, particularly in large-scale indoor environments. RTAB-Map leverages RGB-D sensors and pose-graph optimization via Ceres to generate 3D maps in real time. However, its sequential architecture underutilizes multicore CPUs, creating an opportunity for parallel acceleration. A precise evaluation of map accuracy necessitates generating ground-truth point clouds from SDF-defined geometries and aligning SLAM outputs using ICP and bidirectional distance metrics.

![mappings](/assets/gifs/mapping.gif){: .mx-auto.d-block :}
<small> Demo </small>

**Methodology & Key Contributions**  
- **CPU-Level Parallelization with OpenMP:** Front-end loops for feature extraction and matching, and back-end loops for Jacobian assembly and information-matrix reduction, were parallelized using `#pragma omp parallel for` and thread-local buffers, achieving near-linear scaling up to eight cores.  
- **Descriptor Pipeline Benchmarking:** ORB, SURF, SIFT, and FREAK were systematically swapped into RTAB-Map’s feature pipeline to analyze speed–accuracy trade-offs. SURF and SIFT improved RMSE by over **10%** compared to the default configuration, while ORB delivered the fastest extraction at the cost of increased mean error.  
- **Ground-Truth Generation & Evaluation:** Custom scripts parsed Gazebo’s SDF world to sample dense point clouds from box primitives. Open3D’s ICP alignments with a 1 m threshold and bidirectional nearest-neighbor distances produced mean distance, RMSE, maximum deviation, and Chamfer distance metrics, enabling quantitative comparison of SLAM outputs.  
- **Ceres Solver Configuration:** The optimizer was configured to exploit all available CPU threads (`options.num_threads = omp_get_max_threads()`), reducing average pose-graph solve time from **0.0923 s** to **0.0767 s** while preserving convergence properties.

**Results**  
- **Throughput Improvement:** Overall mapping speed increased by **1.8×**.  
- **Pose-Graph Optimization:** Average solver time decreased from **0.0923 s** to **0.0767 s**.  
- **Feature Accuracy:** SURF achieved a mean error of **0.42 mm**, SIFT **0.57 mm**, and ORB **2.76 mm**.  
- **Scaling Efficiency:** Speedup remained near-linear up to **8 threads**, with marginal gains beyond.

![speedup wm](/assets/img/parallel_slam.png){: .mx-auto.d-block :}
<small> Optimization time of sequential and parallel CERES solver wrt to working memory (WM) size </small>

**Conclusion**  
By fusing CPU-level parallel acceleration with a systematic feature-pipeline evaluation and a rigorous ground-truth framework, this project demonstrates meaningful real-time SLAM improvements. Future efforts can explore GPU offloading, dynamic scene adaptation, and multimodal sensor integration to further enhance mapping speed and robustness.


## PRM-Based Global Body Planner for Quadruped Robot
<a name="planner"></a>
### Individual research + course project: Planning & Decision Making in Robotics

This project developed a **Probabilistic Roadmap (PRM)-based global body planner** for a quadruped robot built atop the **QUAD SDK stack** using **ROS Noetic**. The planner introduced improvements in dynamic obstacle handling, adaptive z-height adjustments, and computational efficiency, addressing limitations in the existing **RRT-Connect-based framework**.

#### Quad SDK

Quad-SDK is an open-source ROS framework for quadruped robots, developed by Carnegie Mellon’s Robomechanics Lab. It provides modular tools for locomotion, control, and planning, supporting both real-world and simulated environments. It includes:

- A **global planner** using **RRT-Connect** for path generation.
- A **local planner**, implemented as a **Nonlinear Model Predictive Controller (NMPC)**, to refine and execute paths.
- A **footstep planner**, responsible for optimizing foot placement in challenging terrains.

#### Shortcomings of the Existing Framework

- **Static z-Height Planning**: Body height adjustments were minimal, limiting adaptability to varying terrain elevations.
- **Limited Handling of Dynamic Obstacles**: Full replanning was required when obstacles moved.
- **Inefficient Map Reuse**: Previously generated roadmaps were discarded during replanning, increasing computational overhead.

#### Project Additions

To address these issues, the project implemented a **Lazy PRM-based planner** that incorporated dynamic roadmap updates and advanced search algorithms for efficient and adaptive pathfinding.

##### Global Planner with Lazy PRM

- Nodes represented feasible body configurations, including z-height variations based on terrain elevation.
- A roadmap was incrementally validated and reused, avoiding redundant computations during replanning.
- Pathfinding was enhanced using optimized search techniques for balancing efficiency and path quality.

##### z-Height Integration

- Integrated adaptive z-height adjustments, enabling the robot to dynamically modify its posture to maintain stability on uneven terrain.

##### Dynamic Obstacle Handling

- Improved responsiveness by updating roadmap edges incrementally, reducing the need for complete replanning when obstacles moved.

##### Dynamic Path Refinement

- Developed a **trajectory smoothing module** to optimize PRM-generated paths, ensuring dynamic feasibility and compatibility with the SDK's NMPC-based local planner.

![obstacle](/assets/gifs/Obstacle.gif){: .mx-auto.d-block :}
<small> Demo in Gazebo </small>

#### Results

- The **PRM-based global planner** improved adaptability to dynamic obstacles and varying terrain compared to the RRT-Connect planner.
- Efficient map reuse and Lazy PRM validation reduced computational overhead, enabling faster responses in changing environments.
- **z-Height adjustments** provided enhanced stability and terrain adaptability, ensuring smooth and reliable navigation.
- The integrated system demonstrated efficient and adaptive performance in simulations involving complex and dynamic scenarios.

This project highlights how incorporating advanced planning frameworks like **Lazy PRM** into modular SDKs can enhance quadruped robots' autonomous navigation capabilities for real-world applications.

[Project Report](/assets/project_reports/Planning_Final_Report.pdf)  
[GitHub Repo](https://github.com/RedTorus/quad-sdk-PlanningProject) 

<a href="#top" class="btn btn-primary">Back to Project Selection</a>

## Robust Control for Low-Mass Quadrotors under Wind Disturbances
<a name="drone"></a>
### Course project: Advanced Control System Integration
This project focused on developing and evaluating robust control strategies for quadrotors operating under wind disturbances, using the Crazyflie 2.0 platform. The drone was modeled with cascaded dynamics, decoupling attitude and position control. Three control algorithms were implemented: Proportional-Integral-Derivative (PID), Linear Quadratic Regulator (LQR), and Sliding Mode Control (SMC). The project followed a simulation-to-hardware pipeline to design, test, and deploy these controllers.
![Drone flying with SMC in z,roll and pitch direction and PID in x,y and yaw](/assets/gifs/SMChover.gif){: .mx-auto.d-block :}
<small> Drone flying with SMC in z,roll and pitch direction and PID in x,y and yaw </small>
### Key Contributions:
1. **Simulation-to-Hardware Pipeline:**  
   The controllers were first tested in a ROS2 and Gazebo simulation environment with wind modeling. This pipeline facilitated the transition to hardware, allowing for iterative tuning and real-world validation.  

2. **Drone Modeling with Cascaded Dynamics:**  
   The quadrotor was modeled with cascaded dynamics, which decoupled attitude control (roll, pitch, yaw) from position control. This approach simplified the design of control algorithms and enhanced stability under disturbances.  
![Closed loop cascaded system block diagram](/assets/img/ClosedLoop.png){: .mx-auto.d-block :}
<small> Closed loop cascaded system block diagram </small>
3. **Controller Designs:**  
   - **PID:** Provided a simple yet effective baseline for trajectory tracking and hover stability.  
   - **LQR:** Demonstrated exceptional robustness to wind disturbances but sacrificed some position accuracy.  
   - **SMC:** Showed high disturbance rejection capabilities but required extensive tuning to minimize chattering effects.

4. **Hardware Implementation:**  
   The Crazyflie quadrotor's firmware was modified to integrate the custom controllers, enabling dynamic selection of PID, LQR, and SMC during flight without reflashing.  

5. **Wind Testing and Real-World Evaluation:**  
   - In simulated and real-world tests, LQR excelled in orientation stability, while PID maintained moderate accuracy in trajectory tracking.  
   - SMC handled extreme disturbances effectively but struggled with trajectory tracking due to chattering.  

6. **Practical Insights:**  
   - The study highlights the trade-offs between controller robustness and precision.  
   - The simulation-to-hardware pipeline proved essential for bridging the sim-to-real gap, accounting for unmodeled dynamics and hardware constraints.

This work demonstrated the feasibility of deploying advanced controllers on low-cost drones, showcasing applications in dynamic environments such as search-and-rescue and industrial inspections.  

[Project Report](/assets/project_reports/AdvControlSysInt_Report.pdf)  
[GitHub Repo](https://github.com/willkraus9/GustGurus-Drone-Project) 

<a href="#top" class="btn btn-primary">Back to Project Selection</a>

---

## RL for Autonomous Humanoid Bi-Manipulation
<a name="boardwalk"></a>
### Internship Summer 2024

During my internship at Boardwalk Robotics Inc., I worked on developing a reinforcement learning (RL) pipeline for bi-manual manipulation tasks. This involved using the Sake Hands manipulators on the upper body of their humanoid robot, Alex. My primary focus was enabling Alex to pick up a book lying flat on a table.

The simulation environment for this project was built using NVIDIA's Isaac Sim, where I implemented the Proximal Policy Optimization (PPO) algorithm, adapted from the SKRL library. My work included defining coordinate frames, applying domain randomization to enhance generalization, and tuning hyperparameters for the policy and value networks. I also designed reward functions, termination conditions, and unit tests to ensure the pipeline’s reliability and effectiveness.

![Isaac Sim training environment](/assets/img/TrainingIsaac.jpg){: .mx-auto.d-block :}
<small> Isaac Sim training environment </small>

One of the key aspects of this project was designing a curriculum that incrementally increased task complexity. To facilitate learning for more challenging tasks, I pre-trained the networks on simpler objectives before gradually introducing harder goals. This approach enabled efficient learning and established a flexible simulation baseline capable of generalizing to objects of different shapes and sizes.

The pipeline has significant potential for sim-to-real transfer, enabling skills learned in simulation to be applied in real-world scenarios. This work lays the groundwork for Alex to perform similar pick-and-place tasks in practical applications.

As an initial project, I trained a Franka Emika Panda arm to open a drawer and move it aside. While a smaller part of my contributions, this task helped me gain familiarity with the tools and techniques necessary for handling more complex tasks with Alex.

This experience enhanced my skills in reinforcement learning, robotics, and algorithm tuning while demonstrating the potential of simulation frameworks to address real-world robotic manipulation challenges.

<a href="#top" class="btn btn-primary">Back to Project Selection</a>

---

## Sketch-to-Image Diffusion Model
<a name="imagediff"></a>
### Course Project: Intro to Deep Learning

This project involved developing a latent diffusion model conditioned on hand-drawn sketches to generate photorealistic images. The model was trained on the **Sketchy Dataset**, which contains photorealistic images paired with corresponding sketches. To optimize training time, the model focused on four classes: tiger, dog, cat, and zebra.

#### Key Contributions:
- **Data Augmentation:**  
  Implemented an edge map-based data augmentation pipeline using a pretrained ResNet model to generate simple sketches from images.  
  - Applied this pipeline to the entirety of the Sketchy Dataset, increasing the number of sketches by 20%.  

- **Latent Space Encoding:**  
  Developed and trained a Variational Autoencoder (VAE) and Autoencoder (AE) to encode both images and sketches into a shared latent space for effective conditioning.  

- **Experimentation with Diffusion Models:**  
  Iteratively experimented with multiple diffusion models and conditioning techniques to refine the final latent diffusion model.
  
![Unet architecture for predicting noise (for 32x32 pixel input in lowest resolution)](/assets/img/Unet.png){: .mx-auto.d-block :}  
<small> Unet architecture for predicting noise (for 32x32 pixel input in lowest resolution) </small>

- **Model Pipelines and Loss Functions:**  
  Designed and experimented with various pipelines and loss functions, including:  
  - **L1:** Mean Squared Error (MSE) loss between actual and predicted noise from the U-Net.  
  - **L2:** Reconstruction loss between the generated and target images.  

  **Key Model Versions:**  
  - **V4:** Trained with a combination of L1 and L2 losses.  
  - **V5:** Trained with L1 loss only.  
  - **V6:** Used class conditioning on both the VAE and U-Net, trained with L1 loss.  

This project demonstrates the iterative development of a robust latent diffusion model, utilizing innovative data augmentation, latent space manipulation, and loss function design to achieve high-quality sketch-to-image synthesis. Each model was trained for 1000 epochs.

![High level network architecture](/assets/img/ArchitectureD.png){: .mx-auto.d-block :}
<small> High level network architecture </small>

![V5 model output for tiger sketch](/assets/img/ResultV5.png){: .mx-auto.d-block :}
<small> V5 model output for tiger sketch </small>

[Project final presentation video](https://www.youtube.com/watch?v=I5AZhSPdTo0)  
[Github Repo](https://github.com/RedTorus/SketchtoImage)

<a href="#top" class="btn btn-primary">Back to Project Selection</a>

---

## Model-based Reinforcement Learning and Transformer Architecture in a Humanoid Robot Environment
<a name="humrl"></a>
### Course Project: Intro to Robot Learning

This project explored integrating transformers into model-based reinforcement learning (RL) for whole-body control in humanoid robots. The primary objective was to replace traditional multi-layer perceptrons (MLPs) with a transformer architecture within the TD-MPC2 framework, enhancing performance and reducing training time.

<table>
  <tr>
    <td style="text-align: center;">
      <img src="/assets/gifs/PureTDMPC.gif" alt="TD-MPC using MLPs" style="width: 300px;">
      <p>TD-MPC using MLPs</p>
    </td>
    <td style="text-align: center;">
      <img src="/assets/gifs/TrafoTDMPC.gif" alt="TD-MPC using decision Transformer" style="width: 300px;">
      <p>TD-MPC using decision Transformer</p>
    </td>
  </tr>
</table>

#### Key Highlights:
- **Transformer Integration:**  
  A decision transformer was used to predict actions, states, rewards, and Q-values in the RL pipeline, replacing MLPs traditionally used in TD-MPC2.  

- **Pretraining on MT30 Dataset:**  
  Pretrained on 345 million transitions across 30 tasks and 11 robot models, the transformer demonstrated improved learning efficiency and smoother motion dynamics.  

- **Performance Improvements:**  
  - 54% reduction in training time with randomly initialized weights.  
  - 27% reduction in training time with pretraining, along with smoother and more natural joint movements.  

- **Task Evaluation:**  
  Evaluated on the HumanoidBench "sit simple" task, which involves challenging contact dynamics. The transformer achieved comparable performance to traditional RL methods with reduced training time and improved motion smoothness.

This work showcases the potential of transformers in robotics, offering better generalization, reduced training time, and effective handling of sequential data, paving the way for more efficient reinforcement learning frameworks in humanoid robots.

[Project Report](/assets/project_reports/RobotLearningFinal_Report.pdf)  
[Project Github](https://github.com/Woodwardbr/16831-project/tree/feature/hf-transformer)

<a href="#top" class="btn btn-primary">Back to Project Selection</a>

## Hand Slapper: A Reaction Time Game
<a name="slapper"></a>
### Course Project: Advanced Mechatronic Design

The **Hand Slapper** is an interactive reaction time game where users attempt to avoid a motor-driven swatter while placing their hand on a sensor-equipped platform. The system integrates precise motor control, real-time feedback, and sensor validation, all managed by an STM32 microcontroller running custom bare-metal firmware. This approach involved writing the firmware from scratch, giving full control over hardware resources without relying on external libraries.
![Demo](/assets/gifs/slapper.gif){: .mx-auto.d-block :}
#### Hardware Design:
- **Actuators and Sensors:**
  - **Motor:** 12V DC motor (Pololu #4680) with encoders, controlled via an L293D motor driver.
  - **IR Sensors (x5):** Detect user finger positions on the platform.
  - **Force Sensitive Resistor (FSR):** Validates hand pressure for game initiation.
  - **Control Buttons:** Start and stop gameplay.
- **Mechanical Components:**
  - 3D-printed motor housing, swatter mount with counterweights, and a laser-cut wooden platform.
- **Microcontroller Usage:**
  - **GPIOs:** ~10 for interfacing sensors, buttons, and the motor driver.
  - **Timers:** Used for PID motor control and randomized delay generation.
  - **Interrupts:** Ensures fast reaction to sensor inputs.

#### Software and Control Logic:
- **Bare-Metal Firmware:** Firmware was written from scratch, managing hardware directly to achieve precise control without overhead from external libraries.
- **PID Motor Control:** Provides precise swatter motion using encoder feedback.
- **Game State Machine:** Manages transitions between idle, gameplay, and scoring states, preventing improper hand placement.
- **Randomized Gameplay:** Introduces dynamic delays to enhance challenge.

This project highlights the STM32 microcontroller's capabilities in real-time mechatronics using bare-metal programming, demonstrating advanced control techniques in a fast-paced reaction-based gaming application.

[Project Report](/assets/project_reports/AdvMechDesignReport.pdf) 

<a href="#top" class="btn btn-primary">Back to Project Selection</a>

## Grasp Optimization from Learning-Based Initial Guess
<a name="thesis"></a>
### Bachelor Thesis at Chair of Information oriented Control (ITR) at TUM
This thesis introduces optimization techniques to refine robotic grasp configurations derived from a reinforcement learning (RL) framework for the Franka Panda robot. The thesis focuses on two critical aspects: **contact position optimization** and **force optimization**, with implementation carried out in the simulation environment MuJoCo. The objective is to enhance grasp stability and efficiency through advanced optimization methods. Simulation tests were performed using a two-finger parallel jaw gripper grasping cubic objects.

#### **Contact Position Optimization**
![Before and after position optimization](/assets/img/graspBA.png){: .mx-auto.d-block :}
<small> Before and after position optimization </small>
- **Objective**: Refine the placement of contact points to achieve force closure and unit frictionless equilibrium (UFE).
- **Methodology**:
  - Developed algorithms to iteratively minimize force and moment residuals, thereby improving contact positions.
  - Utilized local spherical coordinate parameterization for the optimization of contact points on object surfaces:
    - Ensured the gradients of spherical coordinates aligned with surface normals.
    - Addressed edge cases, such as ambiguous coordinate definitions near object edges, by dynamically adjusting spherical parameters.
  - Implemented **force residual control** to align contact normals and reduce force imbalances.
  - Applied **moment residual control** to balance torques and achieve robust positioning.
 
    <table>
     <tr>
       <td style="text-align: center;">
         <img src="/assets/img/ForceRes.png" alt="Force residual control algorithm" style="width: 45%;">
         <p>Force residual control algorithm</p>
       </td>
       <td style="text-align: center;">
         <img src="/assets/img/MomentRes.png" alt="Moment residual control algorithm" style="width: 45%;">
         <p>Moment residual control algorithm</p>
       </td>
     </tr>
 

 <table>
     <tr>
       <td style="text-align: center;">
         <img src="/assets/img/Force.jpeg" alt="Force residual control concept" style="width: 62%;">
         <p>Force residual control concept</p>
       </td>
       <td style="text-align: center;">
         <img src="/assets/img/Moment.jpeg" alt="Moment residual control concept" style="width: 47%;">
         <p>Moment residual control concept</p>
       </td>
     </tr>
  </table>

**Results**:
  - Successfully minimized force and moment residuals for a variety of initial grasp configurations.
  - Achieved **force closure grasps** for all tested cases by reducing force residuals to near-zero values, significantly improving grasp stability.
  - Enhanced torque balance by minimizing moment residuals, resulting in better equilibrium of grasps.
  - Visual and quantitative evaluations showed clear improvements in grasp quality after optimization, demonstrated through simulation results such as reduced contact misalignments and improved contact configurations.
  - Limitations in evaluating grasp quality metrics for smaller objects (e.g., cubes with 0.05 m edge lengths) were identified, as minor object size changes minimally affected grasp metrics.

#### **Force Optimization**

- **Objective**: Minimize grasping forces while maintaining stability and ensuring compliance with friction and torque constraints.
- **Methodology**:
  - Adopted a **Lagrangian optimization approach** to solve a convex quadratic objective function under linearized friction cone constraints.
  - Used **IPOPT (Interior Point Optimizer)** solver for efficient computation of the optimization problem.
  - Linearized the friction cone into pyramidal edges, with increasing precision as the number of edges was scaled up (e.g., 4, 8, 16, 32, 64 edges).
  - Incorporated soft finger contact (SFC) models, extending the optimization to include torque constraints alongside force closure requirements.
- **Results**:
  - Achieved significant reductions in contact forces, with up to a **10% decrease** in required forces after optimization.
  - Demonstrated a strong correlation between the number of linearized friction cone edges and optimized force levels, with higher edge counts leading to improved results.
  - Validated the robustness of the force optimization framework through simulations, achieving stable and efficient grasps with the parallel jaw gripper.

#### **Overall Contributions**

- Successfully integrated RL-based grasp initialization with optimization techniques to enhance grasp quality, stability, and efficiency.
- Provided detailed theoretical and practical solutions for optimizing both contact position and force, leveraging spherical coordinates for position refinement and IPOPT for solving force optimization problems.
- Evaluated the proposed methods through simulations using a parallel jaw gripper, achieving robust and stable grasps even for simple object geometries like cubes.
- Established a scalable framework, laying the groundwork for future extensions to more complex grippers and object geometries.

This thesis represents a significant step toward deploying RL-based grasping frameworks in real-world robotic manipulation tasks, emphasizing stability, adaptability, and computational efficiency.

[Thesis](/assets/KPaul_Bachelorthesis.pdf) 
[Github Repo](https://github.com/RedTorus/Thesis)

<a href="#top" class="btn btn-primary">Back to Project Selection</a>

## Sound Localization and Autonomous Navigation
<a name="slam"></a>
### Course project: Introduction to autonomous systems
This project integrates sound localization, autonomous navigation, and real-time mapping to simulate a rescue scenario. A robot detects sound sources, navigates toward them, and builds a map of its environment. Implemented in ROS2 Humble, the system combines sound signal detection, SLAM, path planning, and collision monitoring.

![Partial map in RVIZ](/assets/img/slam_wrld.png){: .mx-auto.d-block :}
<small> Partial map in RVIZ </small>
#### Key Components:
1. **Sound Source Detection:** Detects and localizes sound sources, providing positional data for navigation.  
2. **Path Planning and Navigation:** Plans efficient paths and dynamically avoids obstacles.  
3. **SLAM and Collision Monitoring:** Constructs a real-time map using Simultaneous Localization and Mapping (SLAM) and ensures safety during navigation with a collision monitoring system.

#### SLAM and Collision Monitoring:
I worked on the SLAM and Navigation components. Key contributions included:  

- **SLAM Configuration:**  
  - Optimized SLAM Toolbox parameters for precise real-time mapping.  
  - Explored alternative SLAM algorithms (e.g., Gmapping, HectorSLAM) and resolved map serialization issues.  

- **Navigation Integration:**  
  - Modified the robot’s SDF file to enable 360° LiDAR perception with an extended range of 16 meters.  
  - Unified SLAM and Gazebo simulation into a single launch file (`Bringup.py`) for seamless operation.  
  - Configured costmaps and goal-following behavior for efficient navigation while maintaining SLAM performance.  

- **Collision Monitoring:**  
  Integrated a collision monitor as a safety layer to prevent collisions. This system intercepted velocity commands and adjusted the robot’s path dynamically, leveraging LiDAR data to avoid obstacles without disrupting the navigation process.  

This work ensured robust navigation and real-time mapping, enabling the robot to autonomously approach sound sources while maintaining environmental awareness.

[Final Presentation](/assets/TAS_presentation.pdf) 

[GitHub Repo](https://github.com/ydschnappi/Sound-localizaiton)

<a href="#top" class="btn btn-primary">Back to Project Selection</a>

## Tour into the picture (implementation)
<a name="cv"></a>
### Course Project: Computer Vision
This project is based on the paper *Tour into the Picture*, which introduces a framework for creating 3D-like navigable environments from a single 2D image by leveraging vanishing points, planar segmentation, and perspective transformations. Inspired by these concepts, the project reconstructs spatial geometry to enable interactive exploration of scenes.

### Key Techniques and Implementation:
![Poster](/assets/img/poster.png){: .mx-auto.d-block :}
<small> Project poster </small>
#### 1. Vanishing Point and Perspective Geometry:
The vanishing point is identified to determine the image's perspective and spatial depth, forming the basis for dividing the image into planar regions.

#### 2. Planar Segmentation and Homography:
The image is segmented into five primary planar surfaces: the back wall, floor, ceiling, and two side walls. Homography transformations are applied to reposition these planes in a 3D space.

#### 3. Foreground-Background Separation:
Users can isolate foreground objects, which are then repositioned within the reconstructed 3D scene while the background is filled in to maintain visual consistency.

#### 4. 3D Box Construction and Interaction:
The segmented planes are assembled into a 3D box-like representation of the scene. Users can rotate, zoom, and navigate through the virtual space, simulating movement within the environment.

![Demov](/assets/gifs/ComV.gif){: .mx-auto.d-block :}
<small> Demo video </small>

This project successfully implements the principles from *Tour into the Picture*, combining computer vision techniques and interactive tools to transform static images into immersive virtual experiences.

[GitHub Repo](https://github.com/RedTorus/CV_G32/tree/main)

<a href="#top" class="btn btn-primary">Back to Project Selection</a>

## Self-Balancing trajectory following Robot
<a name="balancing"></a>
### Course project: Controller Implementation on Microcontrollers

This project implemented a self-balancing robot using a combination of **flatness-based control** and **PID control**. The provided hardware, including a microcontroller, sensors, and motors, was programmed to achieve dynamic balance, resist slight disturbances, and follow predefined trajectories.

![balanc](/assets/gifs/selfB.gif){: .mx-auto.d-block :}

#### Key Components:

1. **Control Strategies:**
   - **Flatness-Based Control:**
     - Used for **trajectory planning** to generate the desired wheel velocity for the robot to follow a specified path.
     - Flatness theory was leveraged to simplify the dynamic system and compute velocity setpoints based on trajectory requirements.

   - **PID Control:**
     - **Velocity Control (Outer Loop):** A PID controller compared the desired velocity (from flatness-based control) with the actual wheel velocity (from encoder feedback) and computed the tilt angle setpoint required to achieve the desired motion.
     - **Angle Control (Inner Loop):** Another PID controller stabilized the robot's tilt by minimizing the error between the tilt angle setpoint (from the outer loop) and the measured tilt angle (from IMU data). This ensured the robot maintained balance while adjusting its tilt for motion.

2. **Cascaded Control Structure:**
   - The system was structured into two control loops:
     - **Outer Loop (Velocity Control):** Processed velocity errors using a PID controller to compute the desired tilt angle.
     - **Inner Loop (Angle Control):** Used a PID controller to actuate motors and stabilize the robot based on the tilt angle setpoint.

3. **Sensor Fusion:**
   - **IMU (Gyroscope and Accelerometer):** Provided angular velocity and tilt data. A **complementary filter** was used to fuse noisy sensor measurements for accurate tilt estimation.
   - **Encoders:** Measured wheel velocity and position, providing feedback for velocity control and trajectory tracking.

4. **Microcontroller and Embedded Techniques:**
   - **Microcontroller:** ATmega32 programmed to handle control logic and hardware interfacing:
     - **Timers:** Generated precise PWM signals to control motor speed and direction.
     - **Interrupts:** Ensured real-time responses to encoder feedback and sensor data.
     - **I2C Communication:** Handled data exchange with the IMU efficiently.
   - **PWM Motor Control:** Actuated motors via an H-bridge driver circuit for smooth and precise motion.

5. **Hardware Details:**
   - Provided hardware included:
     - A robot chassis with two DC motors for movement.
     - An IMU for tilt detection.
     - Wheel encoders for velocity feedback.
     - An H-bridge driver for motor control.


#### Results:
Flatness-based control provided desired velocity setpoints for trajectory tracking. PID controllers managed velocity and tilt angle, enabling the robot to maintain balance while dynamically adjusting to follow trajectories. The robot successfully resisted disturbances and demonstrated precise trajectory following, showcasing the integration of advanced control strategies and embedded systems programming.

<a href="#top" class="btn btn-primary">Back to Project Selection</a>

## Design of a Controller for a Buck-Boost Converter
<a name="buck"></a>
### Course Project: Lab design and practical realization of a voltage converter 
This project focused on designing and implementing controllers for a buck-boost converter, a critical component in power electronics used to efficiently adjust voltage levels. The primary goal was to ensure stable and regulated output voltage despite changes in input voltage or load conditions.
A buck-boost converter without a controller relies entirely on the duty cycle of its switching circuit, making it highly sensitive to variations in operating conditions. Such systems often suffer from significant voltage ripple, slow transient responses, and potential instability, limiting their effectiveness in practical applications.

To address these issues, I began by mathematically modeling the converter to understand its ideal and real-world behaviors. These models were validated through simulations in LTspice, helping refine the design and predict performance.
#### Controller Design
1. **Digital Controller:** Implemented on an Arduino using a PI control algorithm. This controller dynamically adjusted the duty cycle, significantly improving voltage stability and reducing ripple.
2. **Analog Controller:** Designed using operational amplifiers, it included:
   - **Differential Amplifier:** To calculate the error signal between the desired (set) voltage and the measured output voltage.
   - **Inverting Amplifier:** Used for proportional control, scaling the error signal by the proportional gain.
   - **Integrator Circuit:** Designed for integral control to eliminate steady-state error, implemented using an inverting integrator configuration.
   - **Summing Amplifier:** Combined the outputs of the proportional and integral stages to produce the final controller signal.
#### PWM Generator Circuit  
A PWM generator circuit was designed to translate the controller output into a pulse-width-modulated signal. This circuit used a sawtooth waveform generator, built with a NE555 timer IC, and a comparator circuit based on an LM393 IC. The sawtooth signal was compared to the controller output to generate a precise PWM signal, which was then fed into the buck-boost converter for control.
#### Handling Parasitic Capacitances  
Parasitic capacitances in the comparator circuit and other parts of the PWM generator were mitigated by careful component selection and design. A pull-up resistor was added to stabilize the comparator output, and the rise and fall times of the PWM signal were optimized by considering the effects of parasitics. Additionally, decoupling capacitors were placed near power pins of ICs to reduce noise and stabilize the operation.
#### Testing and Results  
Noise filters were incorporated to reduce output voltage ripple further. The system was tested extensively under varying load conditions and input voltages. The controlled system demonstrated:
- Stable output voltage regulation.
- Improved transient response to load changes.
- Reduced ripple compared to the uncontrolled system.
- Higher efficiency with the analog controller.

  ![Analog control of buck-boost converter with changing setpoint(purple) and corresponding PWM signal (yellow)](/assets/gifs/AnalogBuck.gif){: .mx-auto.d-block :}
<small> Analog control of buck-boost converter with changing setpoint(purple) and corresponding PWM signal (yellow) </small>

This project combined theoretical modeling, circuit simulation, and practical implementation, showcasing expertise in power electronics, control systems, and circuit design.

<a href="#top" class="btn btn-primary">Back to Project Selection</a>

## Internship at Siemens Healthineers 
<a name="siemens"></a>
#### Department: SHS DI XP R&D HW MEC
#### Siemens Healthineers (SHS) Diagnostic Imaging (DI) X-ray Products (XP) (R&D) Hardware (HW) Mecatronics (MEC)
#### Focus Area: Electronics and Components in Medical Imaging Systems

During this internship in the R&D Hardware Mechatronics (MEC) division at Siemens Healthineers, I contributed to diagnostics, system integration, automation, and documentation of advanced medical imaging systems. 
#### System Diagnostics and Testing:
##### **Display Control Boards (DCBs):**
- Inspected functional samples for defects, verifying physical stability and firmware functionality through SSH updates.  
- Tested DCBs in mammography systems, ensuring compatibility with production-level hardware.  
##### **Footswitch Diagnostics:**
- Used CAN analyzers and a custom test box to analyze malfunctioning footswitches for mammography systems.  
- Simulated pedal presses in automated testing loops to identify and document fault patterns.  
##### **Ethernet Cards:**
- Verified 150 Ethernet cards for functionality and authenticity.  
- Identified units without warranty and prepared detailed documentation for supplier accountability and reimbursement.  
###### **Aruba Wireless Access Points:**
- Upgraded firmware on Aruba access points and conducted stability tests with wireless X-ray detectors in radiography and fluoroscopy systems.  
- Analyzed system telegrams to ensure reliable communication between access points and system PCs.  
- Compared performance metrics across firmware versions and provided recommendations for further software optimization.  
##### **Mammography Table Diagnostics:**
- Verified the operation of table motors and integrated sensors in mammography systems.  
- Conducted diagnostics to ensure smooth and precise table movements during clinical workflows.  

---

#### Automation and Hardware Integration:

##### **Automated Hardware Testing Using BeagleBone Black:**
- Developed GPIO-based hardware testing solutions for mammography systems using the BeagleBone Black microcontroller.  
- Programmed Python scripts to control and monitor GPIO signals, streamlining test workflows.

##### **VideoManager System Testing:**
- Validated the performance of the VideoManager system, which manages video signals for high-resolution displays in medical imaging.  
- Ensured compatibility and consistent performance with mammography systems under varying scenarios.  

##### **Mammography and Radiography System Workflow Optimization:**
- Conducted workflow tests for the YSIO Xpree radiography system, comparing performance metrics with the YSIO Max.  
- Analyzed time measurements and system logs to identify and resolve workflow inefficiencies.  

#### Documentation and Process Optimization:
- Migrated system documentation into a Markdown-based format using Doxygen, enhancing accessibility and usability.  
- Designed a modular documentation framework for efficient component selection across various business lines.  

This experience provided valuable insights into both the technical and collaborative aspects of medical imaging systems 

<a href="#top" class="btn btn-primary">Back to Project Selection</a>

## Smaller Projects

1. **Offline A-star Planner for Catching a Moving Target in an Arbitrary Map**  
   <a name="astar"></a> Developed an offline A* planner designed to calculate a path to catch a moving target within a given map. The project involved implementing an efficient search algorithm to adapt to dynamic target movement, ensuring the planner could compute feasible paths under different conditions and map configurations.
![pathh](/assets/img/targetpath.png){: .mx-auto.d-block :}
<small> Path of catchhing moving target in 2D map </small>

2. **Sampling-Based Planners for multi DoF Robotic Arm**  
   <a name="PRM"></a> Implemented various sampling-based planners, including RRT, RRT-Connect, RRT*, and PRM, to plan motions for a high-degree-of-freedom robotic arm. The project focused on evaluating and comparing the performance of these planners in terms of efficiency, collision avoidance, and trajectory optimization.
![armm](/assets/gifs/myGif11.gif){: .mx-auto.d-block :}
<small> RRT connect plan for 5 DoF robot arm in 2D world </small>

3. **CMA-ES and Imitation Learning for Bipedal Walker Control**  
   <a name="biped"></a> Applied Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for optimization tasks such as function maximization and control in the CartPole environment. Additionally, used imitation learning techniques for training a bipedal walker (BipedalWalker-v3) in OpenAI Gym. Approaches included regression for learning from expert demonstrations, DAgger for interactive learning, and Diffusion Policy to enhance stability and performance. These methods were aimed at improving the walker’s ability to navigate uneven terrain and maintain balance in a simulated environment, demonstrating the integration of optimization and imitation learning for robotic control tasks. The expert trajectories are supplied from PPO algorithm.
![armm](/assets/gifs/diffusion.gif){: .mx-auto.d-block :}
<small> Walker walking via diffusion policy </small>

<a href="#top" class="btn btn-primary">Back to Project Selection</a>
