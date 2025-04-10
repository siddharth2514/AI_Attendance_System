document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("login-form");
  const regNumber = document.getElementById("regNumber");
  const password = document.getElementById("password");
  const alertBox = document.getElementById("glow-alert");

  function showAlert(message, isSuccess = false) {
    alertBox.textContent = message;
    alertBox.classList.remove("success", "error");
    alertBox.classList.add(isSuccess ? "success" : "error");
    alertBox.style.display = "block";

    setTimeout(() => {
      alertBox.style.display = "none";
    }, 3000);
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();

    const regVal = regNumber.value.trim();
    const passVal = password.value.trim();

    if (regVal === "" || passVal === "") {
      showAlert("Please fill in all fields.", false); // 🔴 Error
    } else {
      showAlert("Login successful! Redirecting...", true); // ✅ Success
      setTimeout(() => {
        window.location.href = "upload.html";
      }, 2000);
    }
  });
});
