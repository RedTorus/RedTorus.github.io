---
title: Engineering Portfolio
subtitle: Click on an image to get started.
---

## RL for Autonomous Humanoid Bi-Manipulation
### Internship summer 2024

During my internship at Boardwalk Robotics Inc. I worked on developing a reinforcement learning (RL) pipeline for bi-manual manipulation tasks. This involved using the Sake Hands manipulators on the upper body of their humanoid robot, Alex. My primary focus was enabling Alex to pick up a book lying flat on a table.

The simulation environment for this project was built using NVIDIA's Isaac Sim, where I implemented the Proximal Policy Optimization (PPO) algorithm, adapted from the SKRL library. My work included defining coordinate frames, applying domain randomization to enhance generalization, and tuning hyperparameters for the policy and value networks. I also designed reward functions, termination conditions, and unit tests to ensure the pipeline’s reliability and effectiveness.

One of the key aspects of this project was designing a curriculum that incrementally increased task complexity. This approach allowed the RL model to learn efficiently and established a flexible simulation baseline capable of generalizing to objects of different shapes and sizes.

The pipeline has significant potential for sim-to-real transfer, enabling skills learned in simulation to be applied in real-world scenarios. This work lays the groundwork for Alex to perform similar pick-and-place tasks in practical applications.

As an initial project, I trained a Franka Emika Panda arm to open a drawer and move it aside. While a smaller part of my contributions, this task helped me gain familiarity with the tools and techniques necessary for handling more complex tasks with Alex.

This experience enhanced my skills in reinforcement learning, robotics, and algorithm tuning while demonstrating the potential of simulation frameworks to address real-world robotic manipulation challenges.

## Text to Image Diffusion Model
