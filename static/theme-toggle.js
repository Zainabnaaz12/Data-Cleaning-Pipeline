// Wait for DOM to load
document.addEventListener("DOMContentLoaded", () => {
  const toggle = document.getElementById("theme-toggle");
  const body = document.body;

  // Load saved theme from localStorage
  const savedTheme = localStorage.getItem("theme");
  if (savedTheme) {
    body.classList.remove("light-theme", "dark-theme");
    body.classList.add(savedTheme);
  }

  // Toggle theme on button click
  if (toggle) {
    toggle.addEventListener("click", () => {
      if (body.classList.contains("light-theme")) {
        body.classList.replace("light-theme", "dark-theme");
        localStorage.setItem("theme", "dark-theme");
      } else {
        body.classList.replace("dark-theme", "light-theme");
        localStorage.setItem("theme", "light-theme");
      }
    });
  }
});
