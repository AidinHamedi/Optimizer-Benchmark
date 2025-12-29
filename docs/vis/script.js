"use strict";

// Close modal on Escape key
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeModal();
  }
});

// Close modal when clicking outside content
document.getElementById("imageModal").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) {
    closeModal();
  }
});

function closeModal() {
  const modal = document.getElementById("imageModal");
  modal.classList.remove("active");
  // Clear content after animation to save memory
  setTimeout(() => {
    modal.querySelector(".modal-content").innerHTML = "";
  }, 200);
}

/**
 * Opens the analysis gallery modal for a specific optimizer and function.
 * @param {string} baseUrl - Base URL for fetching images.
 * @param {string} optimizerId - URL-safe optimizer name.
 * @param {string} functionId - URL-safe function name.
 * @param {string} functionName - Display name of the function.
 * @param {string} ext - Image extension (e.g., .jpg).
 */
function openGallery(baseUrl, optimizerId, functionId, functionName, ext) {
  const modal = document.getElementById("imageModal");
  const contentContainer = modal.querySelector(".modal-content");

  // Define the standard plots available for every benchmark
  const plots = [
    {
      key: "surface",
      title: "2D Surface Trajectory",
      desc: "The optimizer's path on the function contour map. <br><strong>Green</strong>: Start, <strong>Red</strong>: End, <strong>Star</strong>: Global Minimum.",
    },
    {
      key: "dynamics",
      title: "Optimization Dynamics",
      desc: "Time-series view of the Objective Value, Gradient Norm, Step Size, and Path Efficiency over iterations.",
    },
    {
      key: "phase_portrait",
      title: "Phase Portrait (Step vs Grad)",
      desc: "Visualizes the relationship between Step Size and Gradient Norm. <br><strong>Left</strong>: Raw steps. <strong>Right</strong>: Smoothed trend indicating stagnation or instability.",
    },
    {
      key: "update_ratio",
      title: "Effective Update Ratio",
      desc: "The ratio of Step Size to Gradient Norm (||Δx|| / ||∇f||). Ratios consistently > 1.0 suggest aggressive updates.",
    },
    {
      key: "penalty_donut",
      title: "Tuning Cost Breakdown",
      desc: "Distribution of penalty terms used by Optuna during hyperparameter tuning (e.g., boundary violations, inefficiency).",
    },
  ];

  // Generate the HTML for the gallery items
  const itemsHtml = plots
    .map((plot) => {
      const imgUrl = `${baseUrl}/${optimizerId}/${functionId}/${plot.key}${ext}`;
      return `
        <div class="gallery-item">
            <div
                class="gallery-img-container"
                onclick="window.open('${imgUrl}', '_blank')"
                title="Click to open in new tab"
            >
                <img
                    src="${imgUrl}"
                    alt="${plot.title}"
                    loading="lazy"
                    onerror="this.parentNode.innerHTML='<div class=\\'fallback\\'>Image not available</div>'"
                />
                <!-- Arrow Overlay -->
                <div class="gallery-open-btn">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                        <polyline points="15 3 21 3 21 9"></polyline>
                        <line x1="10" y1="14" x2="21" y2="3"></line>
                    </svg>
                </div>
            </div>
            <div class="gallery-text">
                <h3>${plot.title}</h3>
                <p>${plot.desc}</p>
            </div>
        </div>
    `;
    })
    .join("");

  // Populate Modal Structure
  contentContainer.innerHTML = `
    <div class="modal-header">
        <h2>${functionName}</h2>
        <span class="modal-subtitle">Analysis Gallery</span>
        <button class="close-btn" onclick="closeModal()">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M18 6L6 18M6 6l12 12" />
            </svg>
        </button>
    </div>
    <div class="gallery-scroll-container">
        ${itemsHtml}
    </div>
  `;

  // Show Modal
  modal.classList.add("active");
}
