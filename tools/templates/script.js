"use strict";

const PLOT_DEFINITIONS = [
  {
    file: "surface",
    title: "Trajectory on Loss Surface",
    desc: "The <b>Contour Plot</b> shows the path taken by the optimizer (black line) from Start (Green) to Final (Red). <br><b>Analysis:</b> Check if the path is direct or erratic. Does it get stuck in local minima (blue regions) or successfully reach the global minimum (Gold Star)?",
  },
  {
    file: "dynamics",
    title: "Optimization Dynamics",
    desc: "A dashboard of time-series metrics. <br><b>Loss:</b> Should decrease monotonically (or smoothly). <br><b>Gradient Norm:</b> High values indicate steep slopes; dropping to zero indicates a critical point. <br><b>Efficiency:</b> Compares displacement to path length. 1.0 is a perfect straight line.",
  },
  {
    file: "phase_portrait",
    title: "Phase Portrait (Step vs Grad)",
    desc: "Visualizes the optimizer's regime. <br><b>Stagnation:</b> High gradient but small steps (stuck). <br><b>Overshooting:</b> Small gradient but huge steps (momentum instability). <br><b>Convergence:</b> Ideally, both step size and gradient decay toward the bottom-left corner.",
  },
  {
    file: "update_ratio",
    title: "Effective Update Ratio",
    desc: "The ratio of Step Size to Gradient Norm (log scale). <br><b>> 1.0:</b> Aggressive behavior (Momentum/Adaptive). <br><b>< 1.0:</b> Conservative behavior (Standard SGD). <br>Sudden spikes indicate instability.",
  },
  {
    file: "penalty_donut",
    title: "Tuning Cost Breakdown",
    desc: "Shows which penalty terms contributed to the loss during hyperparameter tuning. <br><b>Val Penalty:</b> Final loss value. <br><b>Dist Penalty:</b> Distance from solution. <br><b>Bound Penalty:</b> Going out of bounds.",
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
