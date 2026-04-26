// =============================
// FAVORITES SYSTEM
// With Feedback Loop Integration
// =============================

function addToFavorites(movie, btnId) {

    let favs = JSON.parse(localStorage.getItem("favorites")) || [];

    if (!favs.includes(movie)) {
        favs.push(movie);
        localStorage.setItem("favorites", JSON.stringify(favs));

        // Change button to green "Added" state
        if (btnId) {
            const btn = document.getElementById(btnId);
            if (btn) {
                btn.textContent = "✓ Added to Favorites";
                btn.classList.remove("btn-fav-gold");
                btn.classList.add("btn-fav-added");
                btn.disabled = true;
            }
        }

        // =============================
        // FEEDBACK LOOP
        // =============================
        logPositiveFeedback(movie);

    } else {
        // Already in favorites — show green state
        if (btnId) {
            const btn = document.getElementById(btnId);
            if (btn) {
                btn.textContent = "✓ Added to Favorites";
                btn.classList.remove("btn-fav-gold");
                btn.classList.add("btn-fav-added");
                btn.disabled = true;
            }
        }
    }

    updateFavoritesCount();
}

// =============================
// LOG POSITIVE FEEDBACK
// Sends to backend /feedback/positive
// Used as ground truth for drift detection
// =============================
async function logPositiveFeedback(movie) {
    try {
        await fetch(`/api/feedback/positive?movie=${encodeURIComponent(movie)}`, {
            method: "POST",
        });
        console.log(`Positive feedback logged for: ${movie}`);
    } catch (error) {
        console.warn("Failed to log feedback:", error);
    }
}

// =============================
// UPDATE NAVBAR COUNT
// =============================
function updateFavoritesCount() {
    const favs = JSON.parse(localStorage.getItem("favorites")) || [];
    const counter = document.getElementById("favoritesCounterNavbar");

    if (counter) {
        counter.innerText = favs.length;
    }
}

// run on page load
window.onload = updateFavoritesCount;