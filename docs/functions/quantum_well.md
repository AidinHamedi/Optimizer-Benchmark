# Quantum Well (Ai Generated)

**Function Name:** `QuantumWell`  
**Difficulty:** Deceptive / Complex  
**Type:** Continuous, Differentiable, Highly Multi-Modal

## Description
The **Quantum Well** describes a "Lattice Trap" scenario. It represents a physics-inspired landscape where a global quadratic potential (a harmonic trap) is overlaid with a high-frequency oscillating lattice.

Visually, this looks like a smooth bowl from a distance. However, as the optimizer approaches the center, the landscape reveals itself to be a dense grid of deep holes. The depth of these holes is controlled by a decay factor, meaning the "traps" are deepest exactly where the global minimum is located. This effectively penalizes algorithms that "roll" too fast towards the center, as they may overshoot the central hole and get stuck in an adjacent, slightly shallower hole.

## Mathematical Composition
The function creates a "Cluster" of minima:

1.  **Global Basin:**
    A standard quadratic term ($Scale \times \sum x^2$) provides a global convex trend.
2.  **Lattice Oscillation:**
    A product of cosines ($\prod \cos(freq \cdot x)$) creates a grid structure.
3.  **Spatial Decay:**
    The lattice term is multiplied by an exponential decay ($\exp(-decay \cdot dist)$). This means the multi-modality fades away at long distances (making the problem look easy initially) but becomes intense near the solution.

## Optimization Challenge
*   **Deceptive Gradient:** At the start (far from center), the gradient looks like a simple convex problem.
*   **Precision Docking:** As the optimizer nears $(0,0)$, it enters a field of deep local minima. It must find the specific hole at the origin, which is surrounded by nearly identical (but suboptimal) neighbors.
*   **Barrier Crossing:** Escaping a local minimum near the center requires climbing high barriers relative to the local gradient.
