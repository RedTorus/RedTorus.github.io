---
title: Engineering Portfolio
subtitle: Click on an image to get started.
---

## Robust Control for Low-Mass Quadrotors under Wind Disturbances

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

---

## RL for Autonomous Humanoid Bi-Manipulation

### Internship Summer 2024

During my internship at Boardwalk Robotics Inc., I worked on developing a reinforcement learning (RL) pipeline for bi-manual manipulation tasks. This involved using the Sake Hands manipulators on the upper body of their humanoid robot, Alex. My primary focus was enabling Alex to pick up a book lying flat on a table.

The simulation environment for this project was built using NVIDIA's Isaac Sim, where I implemented the Proximal Policy Optimization (PPO) algorithm, adapted from the SKRL library. My work included defining coordinate frames, applying domain randomization to enhance generalization, and tuning hyperparameters for the policy and value networks. I also designed reward functions, termination conditions, and unit tests to ensure the pipeline’s reliability and effectiveness.

![Isaac Sim training environment](/assets/img/TrainingIsaac.jpg){: .mx-auto.d-block :}
<small> Isaac Sim training environment </small>

One of the key aspects of this project was designing a curriculum that incrementally increased task complexity. To facilitate learning for more challenging tasks, I pre-trained the networks on simpler objectives before gradually introducing harder goals. This approach enabled efficient learning and established a flexible simulation baseline capable of generalizing to objects of different shapes and sizes.

The pipeline has significant potential for sim-to-real transfer, enabling skills learned in simulation to be applied in real-world scenarios. This work lays the groundwork for Alex to perform similar pick-and-place tasks in practical applications.

As an initial project, I trained a Franka Emika Panda arm to open a drawer and move it aside. While a smaller part of my contributions, this task helped me gain familiarity with the tools and techniques necessary for handling more complex tasks with Alex.

This experience enhanced my skills in reinforcement learning, robotics, and algorithm tuning while demonstrating the potential of simulation frameworks to address real-world robotic manipulation challenges.

---

## Text-to-Image Diffusion Model

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

---

## Model-based Reinforcement Learning and Transformer Architecture in a Humanoid Robot Environment

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

## Hand Slapper: A Reaction Time Game
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

## Grasp Optimization from Learning-Based Initial Guess
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
 </table>

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

-**Results**:
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

## Sound Localization and Autonomous Navigation
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

## Design of a Controller for a Buck-Boost Converter
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
