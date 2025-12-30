# Gradient Labyrinth (Ai Generated)

**Function Name:** `GradientLabyrinth`  
**Difficulty:** Extreme  
**Type:** Continuous, Differentiable, Non-Separable, Multi-Modal

## Description
The **Gradient Labyrinth** is a highly complex 2D optimization landscape designed to stress-test an optimizer's ability to navigate non-linear, correlated manifolds. It represents a "worst-case" scenario for coordinate-descent style algorithms and vanilla SGD.

The landscape features a narrow valley floor that follows a sine wave ($v = \sin(u)$). However, the entire coordinate system is rotated by 45 degrees ($\pi/4$), creating strong dependencies between the $x$ and $y$ variables. Furthermore, the "safe" path along the valley floor is shattered by high-frequency cosine bumps, creating a "washboard" effect that traps momentum-deficient optimizers.

## Mathematical Composition
The function is composed of four distinct layers:

1.  **Coordinate Rotation:**
    The inputs $x, y$ are transformed into $u, v$ via a rotation matrix. This ensures that the optimal direction is never aligned with the axes, making variables highly non-separable.
2.  **Twisted Manifold (The Valley):**
    A quadratic penalty creates steep walls ($Wall \times (v - \sin(u))^2$). The valley floor twists following a sine wave rather than a straight line.
3.  **Global Trend:**
    A very weak quadratic bias pulls the optimizer longitudinally towards the origin. This represents a "vanishing gradient" scenario where the directional signal is drowned out by the steep walls.
4.  **Shattered Floor (Traps):**
    A grid of local minima is added to the floor: $Depth \times (1 - \cos(u)\cos(v))$. This mimics a rugged loss landscape where local optima obscure the global solution.

## Optimization Challenge
*   **Ill-Conditioning:** The ratio between the steepness of the walls and the slope of the floor is massive.
*   **Parameter Coupling:** Due to rotation, movement in $x$ requires a precise compensatory movement in $y$ to stay in the valley.
*   **Local Traps:** The floor is not smooth; optimizers must have enough energy (momentum) to hop over ridges but enough damping to stop at the global minimum.

{% include mathjax.html %}
