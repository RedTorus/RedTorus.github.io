---
title: Engineering Portfolio
subtitle: Click on an image to get started.
---

## Robust Control for Low-Mass Quadrotors under Wind Disturbances

This project focused on developing and evaluating robust control strategies for quadrotors operating under wind disturbances, using the Crazyflie 2.0 platform. The drone was modeled with cascaded dynamics, decoupling attitude and position control. Three control algorithms were implemented: Proportional-Integral-Derivative (PID), Linear Quadratic Regulator (LQR), and Sliding Mode Control (SMC). The project followed a simulation-to-hardware pipeline to design, test, and deploy these controllers.

<!--more-->

### Key Contributions:
1. **Simulation-to-Hardware Pipeline:**  
   The controllers were first tested in a ROS2 and Gazebo simulation environment with wind modeling. This pipeline facilitated the transition to hardware, allowing for iterative tuning and real-world validation.  

2. **Drone Modeling with Cascaded Dynamics:**  
   The quadrotor was modeled with cascaded dynamics, which decoupled attitude control (roll, pitch, yaw) from position control. This approach simplified the design of control algorithms and enhanced stability under disturbances.  
![Closed loop cascaded system block diagram](/assets/img/ClosedLoop.png){: .mx-auto.d-block :}
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

<!--more-->

The simulation environment for this project was built using NVIDIA's Isaac Sim, where I implemented the Proximal Policy Optimization (PPO) algorithm, adapted from the SKRL library. My work included defining coordinate frames, applying domain randomization to enhance generalization, and tuning hyperparameters for the policy and value networks. I also designed reward functions, termination conditions, and unit tests to ensure the pipeline’s reliability and effectiveness.

![Isaac Sim training environment](/assets/img/TrainingIsaac.jpg){: .mx-auto.d-block :}

One of the key aspects of this project was designing a curriculum that incrementally increased task complexity. To facilitate learning for more challenging tasks, I pre-trained the networks on simpler objectives before gradually introducing harder goals. This approach enabled efficient learning and established a flexible simulation baseline capable of generalizing to objects of different shapes and sizes.

The pipeline has significant potential for sim-to-real transfer, enabling skills learned in simulation to be applied in real-world scenarios. This work lays the groundwork for Alex to perform similar pick-and-place tasks in practical applications.

As an initial project, I trained a Franka Emika Panda arm to open a drawer and move it aside. While a smaller part of my contributions, this task helped me gain familiarity with the tools and techniques necessary for handling more complex tasks with Alex.

This experience enhanced my skills in reinforcement learning, robotics, and algorithm tuning while demonstrating the potential of simulation frameworks to address real-world robotic manipulation challenges.

---

## Text-to-Image Diffusion Model

### Course Project: Intro to Deep Learning

This project involved developing a latent diffusion model conditioned on hand-drawn sketches to generate photorealistic images. The model was trained on the **Sketchy Dataset**, which contains photorealistic images paired with corresponding sketches. To optimize training time, the model focused on four classes: tiger, dog, cat, and zebra.

<!--more-->

#### Key Contributions:
- **Data Augmentation:**  
  Implemented an edge map-based data augmentation pipeline using a pretrained ResNet model to generate simple sketches from images.  
  - Applied this pipeline to the entirety of the Sketchy Dataset, increasing the number of sketches by 20%.  

- **Latent Space Encoding:**  
  Developed and trained a Variational Autoencoder (VAE) and Autoencoder (AE) to encode both images and sketches into a shared latent space for effective conditioning.  

- **Experimentation with Diffusion Models:**  
  Iteratively experimented with multiple diffusion models and conditioning techniques to refine the final latent diffusion model.
  
![Unet architecture for predicting noise (for 32x32 pixel input in lowest resolution)](/assets/img/Unet.png){: .mx-auto.d-block :}  

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

![V5 model output for tiger sketch](/assets/img/ResultV5.png){: .mx-auto.d-block :}

[Project final presentation video](https://www.youtube.com/watch?v=I5AZhSPdTo0)  
[Github Repo](https://github.com/RedTorus/SketchtoImage)

---

## Model-based Reinforcement Learning and Transformer Architecture in a Humanoid Robot Environment

### Course Project: Intro to Robot Learning

This project explored integrating transformers into model-based reinforcement learning (RL) for whole-body control in humanoid robots. The primary objective was to replace traditional multi-layer perceptrons (MLPs) with a transformer architecture within the TD-MPC2 framework, enhancing performance and reducing training time.

<!--more-->

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

