// =============================
// MAIN CONTROLLER (FASTAPI CONNECT)
// =============================

console.log("JS WORKING");

let animFrameId = null;
let userTouched = false;

async function getRecommendations(movieName) {

    if (!movieName) {
        alert("Please enter a movie name");
        return;
    }

    try {
        const response = await fetch(
            `/api/recommend?movie=${encodeURIComponent(movieName)}`
        );
        
        // Also handle non-OK responses properly
        if (!response.ok) {
            const err = await response.json();
            alert(`Error: ${err.detail || "Something went wrong"}`);
            return;
        }
        
        console.log("STATUS:", response.status);
        const data = await response.json();
        console.log("DATA:", data);

        displayRecommendations(data);

        // Scroll page down to recommendations section
        document.getElementById("recommendations").scrollIntoView({
            behavior: "smooth"
        });

        // Wait for cards to render, then start conveyor scroll
        setTimeout(() => {
            startConveyorScroll();
        }, 1000);

    } catch (error) {
        console.error("Error:", error);
        alert("Backend not running");
    }
}

// =============================
// CONVEYOR BELT SCROLL
// one direction, slow, loops back silently
// =============================
function startConveyorScroll() {
    const wrapper = document.querySelector(".horizontal-scroll-wrapper");
    if (!wrapper) return;

    stopConveyorScroll();
    userTouched = false;

    const speed = 0.8; // px per frame — lower = slower crawl

    function step() {
        if (userTouched) return;

        const maxScroll = wrapper.scrollWidth - wrapper.clientWidth;

        // Silently jump back to start when we reach the end
        if (wrapper.scrollLeft >= maxScroll - 1) {
            wrapper.scrollLeft = 0;
        } else {
            wrapper.scrollLeft += speed;
        }

        animFrameId = requestAnimationFrame(step);
    }

    animFrameId = requestAnimationFrame(step);

    // Pause when user interacts
    wrapper.addEventListener("mousedown", pauseScroll);
    wrapper.addEventListener("touchstart", pauseScroll);
    wrapper.addEventListener("wheel", pauseScroll);

    // Resume after user stops interacting
    wrapper.addEventListener("mouseup", resumeScroll);
    wrapper.addEventListener("touchend", resumeScroll);
}

function pauseScroll() {
    userTouched = true;
    stopConveyorScroll();
}

function resumeScroll() {
    // Small delay so user can finish their manual scroll
    setTimeout(() => {
        userTouched = false;
        startConveyorScroll();
    }, 800); // resumes 0.8s after user lets go
}

function stopConveyorScroll() {
    if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }
}

// =============================
// HANDLE CAROUSEL CLICK
// =============================
function selectMovie(movieName) {
    const input = document.getElementById("searchInput");
    if (input) input.value = movieName;

    getRecommendations(movieName);
}

document.addEventListener("DOMContentLoaded", function () {
    new Swiper(".movieSwiper", {
        slidesPerView: "auto",
        spaceBetween: 20,
        loop: true,
        pagination: {
            el: ".swiper-pagination",
            clickable: true,
        },
        autoplay: {
            delay: 250,
        },
    });
});