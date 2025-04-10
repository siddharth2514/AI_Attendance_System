document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector(".registration-form");

  // Create and append the glowing alert element
  const alertBox = document.createElement("div");
  alertBox.classList.add("glow-alert");
  alertBox.style.display = "none";
  document.body.appendChild(alertBox);

  form.addEventListener("submit", function (e) {
    e.preventDefault();

    const inputs = form.querySelectorAll("input");
    const allFilled = Array.from(inputs).every(input => input.value.trim() !== "");

    if (!allFilled) {
      showAlert("Please fill in all fields.", false); // 🔴 Error
      return;
    }

    // 🌸 Success message then redirect
    showAlert("Registration successful! Redirecting...", true);
    setTimeout(() => {
      window.location.href = "../html/login.html";
    }, 2000);
  });

  function showAlert(message, isSuccess = false) {
    alertBox.textContent = message;
    alertBox.classList.remove("success", "error");
    alertBox.classList.add(isSuccess ? "success" : "error");
    alertBox.style.display = "block";

    setTimeout(() => {
      alertBox.style.display = "none";
    }, 3000);
  }
});
