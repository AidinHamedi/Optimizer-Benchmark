function openModal(src) {
  const modal = document.getElementById('imageModal');
  const modalImg = document.getElementById('modalImage');
  modalImg.src = src;
  modal.classList.add('active');
}

function closeModal() {
  const modal = document.getElementById('imageModal');
  modal.classList.remove('active');
}

document.addEventListener('keydown', function(event) {
  if (event.key === 'Escape') {
    closeModal();
  }
});

document.getElementById('imageModal').addEventListener('click', function(event) {
  if (event.target === this) {
    closeModal();
  }
});
