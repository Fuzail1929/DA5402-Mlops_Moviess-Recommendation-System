// =============================
// FAVORITES SYSTEM
// =============================

function addToFavorites(movie) {

    let favs = JSON.parse(localStorage.getItem("favorites")) || [];

    if (!favs.includes(movie)) {
        favs.push(movie);
        localStorage.setItem("favorites", JSON.stringify(favs));
        alert("Added to favorites");
    } else {
        alert("Already in favorites");
    }

    updateFavoritesCount();
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