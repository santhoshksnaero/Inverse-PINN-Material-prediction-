# Inverse-PINN-Material-prediction for 1D heat flow in a bar 
"Aerospace Digital Twin prototype: Solving the 1D Heat Equation using Physics-Informed Neural Networks (PINNs). Bridges thermodynamics and AI to simulate engine cooling and discover material properties."
# 1D Heat Transfer: Physics-Informed Intelligence
### *Bridging Thermodynamics and Deep Learning for Aerospace Propulsion*


##  The Vision
In high-performance engines like the **Safran LEAP**, understanding heat flow is the difference between mission success and structural failure. This project moves beyond traditional solvers by using **PINNs** (Physics-Informed Neural Networks) to:

1.  **Simulate:** Predict thermal distribution along an aerospace component.
2.  **Discover:** Reverse-engineer unknown material properties ($\alpha$) from sparse sensor data.
3.  **Optimize:** Rapidly iterate on brain architectures (Hyperparameter tuning) to find the "Sweet Spot" for physics accuracy.

##  Physics Core
This model enforces the **1D Heat Diffusion Equation**:
$$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$
Function can be altered by the user as per the trend observed or required

Unlike standard AI, this "Brain" doesn't just guess—it **obeys the Laws of Thermodynamics** by minimizing the physical residual during training.

##  Key Milestones
- [x] **Phase 1:** Forward Simulation (Modeling heat flow over time).
- [x] **Phase 2:** Inverse Problem Solving (Discovering $\alpha \approx 0.3989112675189972 $ from random sensor data).The random sensor data can be altered by the user as per the observed datatrend or new mathematical functions
- [x] **Phase 3:** Architecture Study (Analyzing "Overfitting Cliffs" in 20x100 neuron networks).

---
*Developed as part of a professional aerospace AI portfolio targeting next-gen propulsion design.*
