// /static/js/animations.js

window.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.querySelector("form");
    const overlay = document.getElementById("analyzing-overlay");
  
    if (uploadForm && overlay) {
      uploadForm.addEventListener("submit", () => {
        overlay.style.display = "flex";
      });
    }
});
