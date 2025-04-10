document.addEventListener("DOMContentLoaded", () => {
    const holoRing = document.querySelector(".holo-lines");
    let ringX = 0, ringY = 0;
    let mouseX = 0, mouseY = 0;
    let lastTime = 0;

    document.addEventListener("mousemove", (event) => {
        mouseX = event.clientX;
        mouseY = event.clientY;
    });

    function animateRing(time) {
        const delta = time - lastTime;
        lastTime = time;

        const speed = 0.04; // smaller = smoother & more trailing
        ringX += (mouseX - ringX) * speed;
        ringY += (mouseY - ringY) * speed;

        const pulse = 1 + Math.sin(time * 0.005) * 0.15;

        holoRing.style.transform = `
            translate(${ringX - holoRing.offsetWidth / 2}px, 
                      ${ringY - holoRing.offsetHeight / 2}px)
            scale(${pulse})
        `;

        requestAnimationFrame(animateRing);
    }

    requestAnimationFrame(animateRing);
});
