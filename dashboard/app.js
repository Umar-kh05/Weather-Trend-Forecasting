// dashboard/app.js
// Interactive features for the executive dashboard

document.addEventListener("DOMContentLoaded", () => {
  console.log("PM Accelerator Weather Trend Forecasting Dashboard Initialized.");

  // Smooth scrolling for navigation links
  document.querySelectorAll('nav a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
      }
    });
  });

  // Image modal preview
  const images = document.querySelectorAll('.card-img-wrap img');
  const modal = document.createElement('div');
  modal.style.position = 'fixed';
  modal.style.top = '0';
  modal.style.left = '0';
  modal.style.width = '100vw';
  modal.style.height = '100vh';
  modal.style.backgroundColor = 'rgba(15, 23, 42, 0.95)';
  modal.style.backdropFilter = 'blur(8px)';
  modal.style.display = 'none';
  modal.style.justifyContent = 'center';
  modal.style.alignItems = 'center';
  modal.style.zIndex = '9999';
  modal.style.cursor = 'zoom-out';

  const modalImg = document.createElement('img');
  modalImg.style.maxWidth = '90vw';
  modalImg.style.maxHeight = '90vh';
  modalImg.style.borderRadius = '12px';
  modalImg.style.boxShadow = '0 20px 50px rgba(0,0,0,0.8)';
  modal.appendChild(modalImg);
  document.body.appendChild(modal);

  images.forEach(img => {
    img.style.cursor = 'zoom-in';
    img.addEventListener('click', () => {
      modalImg.src = img.src;
      modal.style.display = 'flex';
    });
  });

  modal.addEventListener('click', () => {
    modal.style.display = 'none';
  });
});
