---
title: Engineering Portfolio
subtitle: Click on an image to get started.
---

## RL for Autonomous Humanoid Bi-Manipulation
### Internship summer 2024

During my internship at Boardwalk Robotics Inc. I worked on developing a reinforcement learning (RL) pipeline for bi-manual manipulation tasks. This involved using the Sake Hands manipulators on the upper body of their humanoid robot, Alex. My primary focus was enabling Alex to pick up a book lying flat on a table.

The simulation environment for this project was built using NVIDIA's Isaac Sim, where I implemented the Proximal Policy Optimization (PPO) algorithm, adapted from the SKRL library. My work included defining coordinate frames, applying domain randomization to enhance generalization, and tuning hyperparameters for the policy and value networks. I also designed reward functions, termination conditions, and unit tests to ensure the pipeline’s reliability and effectiveness.

![Isaac Sim training environment](/assets/img/TrainingIsaac.jpg){: .mx-auto.d-block :}

One of the key aspects of this project was designing a curriculum that incrementally increased task complexity. To facilitate learning for more challenging tasks, I pre-trained the networks on simpler objectives before gradually introducing harder goals. This approach enabled efficient learning and established a flexible simulation baseline capable of generalizing to objects of different shapes and sizes.

The pipeline has significant potential for sim-to-real transfer, enabling skills learned in simulation to be applied in real-world scenarios. This work lays the groundwork for Alex to perform similar pick-and-place tasks in practical applications.

As an initial project, I trained a Franka Emika Panda arm to open a drawer and move it aside. While a smaller part of my contributions, this task helped me gain familiarity with the tools and techniques necessary for handling more complex tasks with Alex.

This experience enhanced my skills in reinforcement learning, robotics, and algorithm tuning while demonstrating the potential of simulation frameworks to address real-world robotic manipulation challenges.

## Text to Image Diffusion Model
### course project Intro to Deep Learning
This project involved developing a latent diffusion model conditioned on hand-drawn sketches to generate photorealistic images. The model was trained on the **Sketchy Dataset**, which contains photorealistic images paired with corresponding sketches. To optimize training time, the model focused on four classes: tiger, dog, cat, and zebra.

### Key Contributions:
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
