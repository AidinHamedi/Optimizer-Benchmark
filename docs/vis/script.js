"use strict";

/**
 * Image Modal Handling
 */
const modal = document.getElementById("imageModal");
const modalImg = document.getElementById("modalImage");

function openModal(src) {
  if (!modal || !modalImg) return;

  modalImg.src = src;
  modal.classList.add("active");

  // Lock body scroll
  document.body.style.overflow = "hidden";
}

function closeModal() {
  if (!modal) return;

  modal.classList.remove("active");

  // Unlock body scroll
  document.body.style.overflow = "";

  // Clear src after transition to prevent flicker on next open
  setTimeout(() => {
    if (modalImg) modalImg.src = "";
  }, 200);
}

// Event Listeners
document.addEventListener("keydown", function (event) {
  if (event.key === "Escape") {
    closeModal();
  }
});

if (modal) {
  modal.addEventListener("click", function (event) {
    if (event.target === this) {
      closeModal();
    }
  });
}
