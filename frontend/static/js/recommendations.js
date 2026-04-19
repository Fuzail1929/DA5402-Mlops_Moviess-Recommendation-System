// =============================================
// TMDB CONFIG - paste your API key here
// =============================================

const TMDB_API_KEY = '8265bd1679663a7ea12ac168da84d2e8'; // 🔑 Replace with your key
const TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w300";

// =============================================
// Fetch poster from TMDB by movie title
// =============================================
async function fetchPosterFromTMDB(title) {
    try {
        const url = `https://api.themoviedb.org/3/search/movie?api_key=${TMDB_API_KEY}&query=${encodeURIComponent(title)}&include_adult=false`;
        const res = await fetch(url);
        const data = await res.json();
        if (data.results && data.results.length > 0 && data.results[0].poster_path) {
            return TMDB_IMAGE_BASE + data.results[0].poster_path;
        }
    } catch (e) {
        console.warn("TMDB fallback failed for:", title);
    }
    return null;
}

// =============================================
// DISPLAY RECOMMENDATIONS
// =============================================
async function displayRecommendations(data) {

    const container = document.getElementById("recommendationsContainer");
    container.innerHTML = "";

    if (data.error) {
        container.innerHTML = `<p style="color:white; padding:20px;">${data.error}</p>`;
        return;
    }

    // Update results count
    const resultsCount = document.getElementById("resultsCount");
    if (resultsCount) resultsCount.textContent = data.recommendations.length;

    // Build all cards first (with placeholder), then fill posters
    const cardElements = [];

    data.recommendations.forEach((movie, index) => {
        const overview = movie.overview ? movie.overview.slice(0, 80) + '...' : 'No description available...';
        const rating = movie.rating || 'N/A';
        const safeTitle = movie.title.replace(/'/g, "\\'");

        const card = document.createElement("div");
        card.className = "movie-card-wrapper";

        card.innerHTML = `
            <div class="glass-card movie-rec-card">

                <div class="movie-poster-wrapper">
                    <img 
                        id="poster-${index}"
                        src="https://via.placeholder.com/300x450/1e293b/6366f1?text=Loading..."
                        class="movie-poster"
                        alt="${movie.title}"
                    >
                </div>

                <div class="movie-details">
                    <h5 class="movie-title">${movie.title}</h5>
                    <p class="movie-meta">⭐ ${rating}</p>
                    <p class="movie-overview-text">${overview}</p>
                    <button class="btn btn-sm btn-primary w-100 mt-2"
                        onclick="addToFavorites('${safeTitle}')">
                        ❤️ Add to Favorites
                    </button>
                </div>

            </div>
        `;

        // Staggered fade-in animation
        card.style.opacity = "0";
        card.style.transform = "translateY(20px)";
        card.style.transition = `all 0.4s ease ${index * 80}ms`;

        container.appendChild(card);
        cardElements.push({ movie, index });

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                card.style.opacity = "1";
                card.style.transform = "translateY(0)";
            });
        });
    });

    // Now load posters - use backend URL if valid, else fetch from TMDB
    cardElements.forEach(async ({ movie, index }) => {
        const imgEl = document.getElementById(`poster-${index}`);
        if (!imgEl) return;

        let posterUrl = movie.poster;

        // Check if poster from backend is actually valid
        const isValidUrl = posterUrl &&
            posterUrl.startsWith("http") &&
            !posterUrl.includes("placeholder") &&
            !posterUrl.includes("None") &&
            !posterUrl.includes("null");

        if (isValidUrl) {
            // Test if the image actually loads
            const testImg = new Image();
            testImg.onload = () => { imgEl.src = posterUrl; };
            testImg.onerror = async () => {
                // Backend URL is broken — fetch from TMDB
                const tmdbPoster = await fetchPosterFromTMDB(movie.title);
                imgEl.src = tmdbPoster || "https://via.placeholder.com/300x450/1e293b/6366f1?text=No+Poster";
            };
            testImg.src = posterUrl;
        } else {
            // No poster from backend — fetch from TMDB
            const tmdbPoster = await fetchPosterFromTMDB(movie.title);
            imgEl.src = tmdbPoster || "https://via.placeholder.com/300x450/1e293b/6366f1?text=No+Poster";
        }
    });
}