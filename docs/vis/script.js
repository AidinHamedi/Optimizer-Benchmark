"use strict";

const PLOT_DEFINITIONS = [
  {
    file: "surface",
    title: "Trajectory on Loss Surface",
    desc: "The <b>Contour Plot</b> visualizes the path taken by the optimizer (black line) from Start (Green) to Final (Red). <br><b>Analysis:</b> Observe the trajectory's smoothness. Does it head straight for the solution, or does it oscillate? Does it get trapped in local basins or successfully reach the global minimum (Gold Star)?",
  },
  {
    file: "dynamics",
    title: "Optimization Dynamics",
    desc: "A dashboard of time-series metrics. <br><b>Loss:</b> Convergence progress over time. <br><b>Efficiency (Bottom-Right):</b> Compares <b>Net Displacement</b> (Solid Line) vs <b>Total Path</b> (Dotted Line). A large gap between these lines indicates 'wasted effort' (orbiting or zigzagging without gaining ground).",
  },
  {
    file: "phase_portrait",
    title: "Phase Portrait (Micro vs Macro)",
    desc: "A side-by-side view of the optimizer's regime (Step Size vs Gradient Norm). <br><i>Note ⚠️:</i> This plot should be interpreted <b>with a grain of salt</b>, as it is a noisy and only weakly reliable metric, best used for qualitative intuition rather than precise conclusions. <br><b>Left (Raw):</b> Shows high-frequency jitter and instability. <br><b>Right (Smooth):</b> Shows the overall flow/trend. <br><b>Diagonal Line:</b> The 1:1 reference. Points above are <b>Aggressive</b> (Momentum/Adaptive), points below are <b>Conservative</b> (SGD).",
  },
  {
    file: "update_ratio",
    title: "Effective Update Ratio",
    desc: "The ratio of Step Size to Gradient Norm (log scale). <br><b>> 1.0:</b> Aggressive behavior (Momentum or Adaptive scaling). <br><b>< 1.0:</b> Conservative behavior (Standard Gradient Descent). <br><b>Spikes:</b> Often indicate instability or escaping a local minimum.",
  },
  {
    file: "penalty_donut",
    title: "Tuning Cost Breakdown",
    desc: "Breakdown of the weighted penalty terms used to score the optimizer during tuning.",
  },
];

const modal = document.getElementById("imageModal");
const modalContent = document.querySelector(".modal-content");

/**
 * Opens the detailed gallery modal for a specific function.
 */
function openGallery(baseUrl, optId, funcId, funcName, ext) {
  if (!modal) return;

  // Clear previous content
  modalContent.innerHTML = `
    <button class="close-btn" onclick="closeModal()">×</button>
    <div class="modal-header">
        <h2>${funcName}</h2>
        <span class="modal-subtitle">Analysis Gallery</span>
    </div>
    <div class="gallery-scroll-container"></div>
  `;

  const container = modalContent.querySelector(".gallery-scroll-container");

  PLOT_DEFINITIONS.forEach((plot) => {
    const imgUrl = `${baseUrl}/${optId}/${funcId}/${plot.file}${ext}`;

    const item = document.createElement("div");
    item.className = "gallery-item";
    item.innerHTML = `
        <div class="gallery-img-container">
            <img src="${imgUrl}" loading="lazy" alt="${plot.title}" onerror="this.parentElement.innerHTML='<div class=fallback>Image not found</div>'">
        </div>
        <div class="gallery-text">
            <h3>${plot.title}</h3>
            <p>${plot.desc}</p>
        </div>
    `;
    container.appendChild(item);
  });

  modal.classList.add("active");
  document.body.style.overflow = "hidden";
}

function closeModal() {
  if (!modal) return;
  modal.classList.remove("active");
  document.body.style.overflow = "";
}

// Event Listeners
document.addEventListener("keydown", function (event) {
  if (event.key === "Escape") closeModal();
});

if (modal) {
  modal.addEventListener("click", function (event) {
    if (event.target === this) closeModal();
  });
}
