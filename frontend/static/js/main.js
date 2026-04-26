// =============================
// MAIN CONTROLLER
// =============================
console.log("JS WORKING");

let animFrameId = null;
let userTouched = false;

// =============================
// GET RECOMMENDATIONS
// =============================
async function getRecommendations(movieName) {
    if (!movieName) {
        alert("Please enter a movie name");
        return;
    }

    try {
        const response = await fetch(
            `/api/recommend?movie=${encodeURIComponent(movieName)}`
        );

        if (!response.ok) {
            const err = await response.json();
            alert(`Error: ${err.detail || "Something went wrong"}`);
            return;
        }

        const data = await response.json();
        displayRecommendations(data);

        document.getElementById("recommendations").scrollIntoView({
            behavior: "smooth"
        });

        setTimeout(() => { startConveyorScroll(); }, 1000);

    } catch (error) {
        console.error("Error:", error);
        alert("Backend not running");
    }
}

// =============================
// GENRE EXPLORE
// Sends genre directly to backend
// =============================
async function searchByGenre(genre) {
    const input = document.getElementById("searchInput");
    if (input) input.value = genre.charAt(0).toUpperCase() + genre.slice(1) + " movies";

    try {
        const response = await fetch(
            `/api/recommend?movie=${encodeURIComponent(genre)}`
        );

        if (!response.ok) {
            const err = await response.json();
            alert(`No movies found for: ${genre}`);
            return;
        }

        const data = await response.json();
        displayRecommendations(data);

        document.getElementById("recommendations").scrollIntoView({
            behavior: "smooth"
        });

        setTimeout(() => { startConveyorScroll(); }, 1000);

    } catch (error) {
        console.error("Genre search error:", error);
        alert("Backend not running");
    }
}

// =============================
// SWITCH VIEW (grid/list)
// =============================
function switchView(viewType) {
    const gridBtn = document.querySelector('[data-view="grid"]');
    const listBtn = document.querySelector('[data-view="list"]');
    const container = document.getElementById("recommendationsContainer");

    if (!container) return;

    if (viewType === "grid") {
        container.classList.remove("list-view");
        container.classList.add("grid-view");
        if (gridBtn) { gridBtn.classList.add("active"); }
        if (listBtn) { listBtn.classList.remove("active"); }
    } else {
        container.classList.remove("grid-view");
        container.classList.add("list-view");
        if (listBtn) { listBtn.classList.add("active"); }
        if (gridBtn) { gridBtn.classList.remove("active"); }
    }
}

// =============================
// CONVEYOR BELT SCROLL
// =============================
function startConveyorScroll() {
    const wrapper = document.querySelector(".horizontal-scroll-wrapper");
    if (!wrapper) return;

    stopConveyorScroll();
    userTouched = false;

    const speed = 0.8;

    function step() {
        if (userTouched) return;
        const maxScroll = wrapper.scrollWidth - wrapper.clientWidth;
        if (wrapper.scrollLeft >= maxScroll - 1) {
            wrapper.scrollLeft = 0;
        } else {
            wrapper.scrollLeft += speed;
        }
        animFrameId = requestAnimationFrame(step);
    }

    animFrameId = requestAnimationFrame(step);

    wrapper.addEventListener("mousedown", pauseScroll);
    wrapper.addEventListener("touchstart", pauseScroll);
    wrapper.addEventListener("wheel", pauseScroll);
    wrapper.addEventListener("mouseup", resumeScroll);
    wrapper.addEventListener("touchend", resumeScroll);
}

function pauseScroll() {
    userTouched = true;
    stopConveyorScroll();
}

function resumeScroll() {
    setTimeout(() => {
        userTouched = false;
        startConveyorScroll();
    }, 800);
}

function stopConveyorScroll() {
    if (animFrameId) {
        cancelAnimationFrame(animFrameId);
        animFrameId = null;
    }
}

// =============================
// CAROUSEL CLICK
// =============================
function selectMovie(movieName) {
    const input = document.getElementById("searchInput");
    if (input) input.value = movieName;
    getRecommendations(movieName);
}

// =============================
// SWIPER INIT
// =============================
document.addEventListener("DOMContentLoaded", function () {
    new Swiper(".movieSwiper", {
        slidesPerView: "auto",
        spaceBetween: 20,
        loop: true,
        pagination: { el: ".swiper-pagination", clickable: true },
        autoplay: { delay: 2500 },
    });
});