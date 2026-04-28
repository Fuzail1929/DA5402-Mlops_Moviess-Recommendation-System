# 🎬 CineMatch — MLOps Movie Recommendation System

[![CI/CD](https://github.com/Fuzail1929/DA5402-Mlops_Moviess-Recommendation-System/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/Fuzail1929/DA5402-Mlops_Moviess-Recommendation-System/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![MLflow](https://img.shields.io/badge/MLflow-3.1.4-orange)
![Airflow](https://img.shields.io/badge/Airflow-2.9.3-red)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

> An AI-powered movie recommendation system with a full MLOps pipeline — automated training, drift detection, real-time monitoring, and CI/CD.

## Screenshots

<p align="center">
  <img src="Screenshots/Home_page.png" width="80%" alt="Home Page"/>
  <br/><em>Home Page</em>
</p>

<p align="center">
  <img src="Screenshots/Search_bar.png" width="80%" alt="Search Bar"/>
  <br/><em>Search Bar</em>
</p>

<p align="center">
  <img src="Screenshots/Result.png" width="80%" alt="Search Results"/>
  <br/><em>Movie Recommendations</em>
</p>

<p align="center">
  <img src="Screenshots/Airflow_dag_run.png" width="80%" alt="Airflow dag"/>
  <br/><em>Airflow dag Successful run</em>
</p>


---

##  Overview

CineMatch takes a movie title as input and returns 10 similar movie recommendations using a TF-IDF + Cosine Similarity model trained on the TMDB 5000 Movies dataset. The system is built with production-grade MLOps practices:

- **Online (real-time) serving** via FastAPI REST API
- **Automated retraining** via Apache Airflow when data drift is detected
- **Full experiment tracking** via MLflow with model registry
- **Real-time monitoring** via Prometheus + Grafana (24-panel dashboard)
- **Data versioning** via DVC + Git
- **CI/CD** via GitHub Actions

---

##  Architecture

```
User → Nginx (Frontend:5050) → FastAPI (Backend:8001) → ML Pipeline
                                      ↓                      ↓
                               TMDB API              MLflow (5001)
                                      ↓                      ↓
                            Feedback Loop            Airflow (8080)
                                      ↓                      ↓
                          Prometheus (9090) → Grafana (3001)
```

All services run in Docker containers on a shared `cinematch-network` bridge network.

---

##  Features

### Frontend
- 🎨 Dark-themed responsive UI built with HTML/CSS/JavaScript
- 🔍 Smart search — movie title, genre, actor name, or character name (e.g. 'Loki', 'Tom Hanks')
- 🎠 Auto-scrolling movie carousel
- 🎭 Genre browsing — direct genre search (Action, Sci-Fi, Drama, Horror, Comedy, etc.)
- ❤️ Favorites system with navbar counter
- 🖼️ Movie cards with TMDB posters, ratings, and overviews

### Backend
- ⚡ FastAPI REST API with 8 endpoints
- 📊 20+ Prometheus metrics instrumented
- 🔄 Feedback loop — every search logged as ground truth
- 📈 Data drift detection — auto-triggers retraining
- 🔁 Automatic rollback if new model underperforms
- 🏥 Health checks at `/health` and `/ready`

### ML Pipeline
- 🤖 TF-IDF vectorizer with 7000 features, bigrams, sublinear TF
- 📐 Cosine similarity with sparse matrix optimization (60-80% memory reduction)
- 🧪 Feature impact analysis (ablation study per feature)
- 📦 MLflow Model Registry with Production/Staging stages
- 🗃️ Feature store with versioned baselines and importance scores

### MLOps
- 🔬 MLflow experiment tracking — params, metrics, artifacts
- 📅 Airflow DAGs — daily preprocessing, weekly training
- 🔄 DVC data versioning — v1/v2/v3 dataset versions
- 🚨 Prometheus alerts — latency, error rate, drift, CPU, memory
- 🔧 GitHub Actions CI/CD — test, validate, build on every push

---

##  Tech Stack

| Category | Technology |
|----------|-----------|
| Frontend | Nginx, HTML5, CSS3, JavaScript, Bootstrap 5, Swiper.js |
| Backend | FastAPI, Python 3.11, Uvicorn |
| ML Model | TF-IDF (scikit-learn), Cosine Similarity, SciPy Sparse |
| Experiment Tracking | MLflow 3.1.4 |
| Orchestration | Apache Airflow 2.9.3 |
| Data Versioning | DVC 3.66, Git |
| Monitoring | Prometheus, Grafana, Node Exporter, cAdvisor |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| External API | TMDB (The Movie Database) |

---

##  Project Structure

```
AI-Project/
├── backend/
│   ├── main.py                    # FastAPI app — all endpoints
│   └── api/
├── frontend/
│   ├── templates/index.html       # Main UI
│   └── static/
│       ├── css/style.css
│       └── js/
│           ├── main.js            # Search & carousel logic
│           ├── recommendations.js # Movie card rendering
│           └── favorites.js       # Favorites + feedback logging
├── ml_pipeline/
│   ├── train.py                   # Full training pipeline
│   ├── recommend.py               # Inference engine
│   ├── preprocess.py              # Feature engineering
│   ├── data_loader.py             # Data validation & loading
│   ├── model/                     # Trained artifacts (DVC tracked)
│   │   ├── movies.pkl
│   │   ├── similarity.pkl
│   │   ├── similarity_sparse.pkl
│   │   ├── vectorizer.pkl
│   │   └── baseline_statistics.json
│   └── feature_store/             # Versioned features
├── dags/
│   ├── cinematch_dag.py           # Daily preprocessing DAG
│   └── cinematch_training_dag.py  # Weekly training DAG
├── config/
│   ├── prometheus.yml             # Prometheus scrape config
│   ├── alerts.yml                 # Alert rules
│   └── grafana/
│       ├── datasources_prometheus.yml
│       ├── grafana_dashboard_provider.yml
│       └── cinematch_dashboard.json
├── data/                          # DVC tracked datasets
│   ├── tmdb_5000_movies.csv.dvc
│   └── tmdb_5000_credits.csv.dvc
├── feedback_logs/                 # Mounted from container
│   ├── search_feedback.csv        # Every search logged
│   ├── positive_feedback.csv      # Favorites logged
│   └── drift_report.json         # Latest drift report
├── .github/workflows/ci_cd.yml   # GitHub Actions CI/CD
├── dvc.yaml                       # DVC pipeline definition
├── MLproject                      # MLflow project file               
├── docker-compose.yaml            # All 10 services
├── Dockerfile.backend
├──Documentation/ 
│   ├── Cinematch_Final_Report.pdf               
│   ├── architecture_diagram.svg
│   ├── Cinematch_HLD.pdf
│   ├── test_plan.pdf
│   ├── Usermanual_CInematch
│   └── Cinematch_LLD.pdf
├── Readme.md
├── Dockerfile.frontend
└── requirements.txt    
```

---

##  Quick Start

### Prerequisites
- Docker Desktop installed and running
- TMDB API key (free at https://www.themoviedb.org/settings/api)

### 1. Clone the repository
```bash
git clone https://github.com/Fuzail1929/DA5402-Mlops_Moviess-Recommendation-System.git
cd DA5402-Mlops_Moviess-Recommendation-System
```

### 2. Set up environment
```bash
cp .env.example .env
# Edit .env and add your TMDB_API_KEY
```

### 3. Pull data with DVC
```bash
pip install dvc
dvc pull
```

### 4. Start all services
```bash
docker compose up -d
```

### 5. Wait ~60 seconds then open
```
http://localhost:5050
```

---

##  Services & URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| CineMatch App | http://localhost:5050 | — |
| FastAPI Backend | http://localhost:8001 | — |
| FastAPI Docs | http://localhost:8001/docs | — |
| MLflow UI | http://localhost:5001 | — |
| Airflow UI | http://localhost:8080 | admin / cinematch123 |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3001 | admin / cinematch123 |
| Node Exporter | http://localhost:9100 | — |
| cAdvisor | http://localhost:8081 | — |

---

##  API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/recommend?movie=X` | Get 10 movie recommendations |
| GET | `/health` | Service health status |
| GET | `/ready` | Readiness check (model + API key) |
| GET | `/metrics` | Prometheus metrics |
| GET/POST | `/feedback/positive?movie=X` | Log positive feedback (favorites) |
| GET | `/feedback` | Feedback statistics |
| GET | `/drift?auto_retrain=false` | Run drift detection |
| GET/POST | `/retrain?reason=manual` | Trigger Airflow retraining DAG |

### Example
```bash
# Get recommendations
curl "http://localhost:8001/recommend?movie=Inception"

# Check drift
curl "http://localhost:8001/drift"

# Trigger retraining
curl -X POST "http://localhost:8001/retrain?reason=demo"
```

---

##  ML Pipeline

### Model
- **Algorithm**: TF-IDF Vectorizer + Cosine Similarity
- **Features**: Overview(1x) + Genres(2x) + Keywords(1x) + Cast(2x) + Director(3x)
- **Vocabulary**: 7000 features, bigrams, sublinear TF
- **Optimization**: Sparse similarity matrix (top-20 per movie, 60-80% memory reduction)
- **Evaluation**: Genre match rate (target ≥ 50%, achieved ~78%)

### Training DAG (Weekly)
```
validate_data → run_training → evaluate_model → check_model_quality
                                                        ↓
                                            promote_model OR rollback_model
                                                        ↓
                                               notify_completion
```

### Preprocessing DAG (Daily)
```
load_data → eda → preprocess
```

### Trigger Training Manually
```bash
# Via Airflow UI at http://localhost:8085
# Or via API:
curl -X POST "http://localhost:8001/retrain?reason=manual"
```

---

##  Monitoring

### Prometheus Metrics (20+)
| Metric | Type | Description |
|--------|------|-------------|
| `cinematch_requests_total` | Counter | Total API requests |
| `cinematch_request_latency_seconds` | Histogram | Request latency |
| `cinematch_inference_latency_ms` | Gauge | End-to-end inference latency |
| `cinematch_recommendations_total` | Counter | Recommendations by status |
| `cinematch_drift_detected` | Gauge | 1 if drift detected |
| `cinematch_retraining_triggered_total` | Counter | Auto-retraining count |
| `cinematch_feedback_total` | Counter | User feedback logged |
| `cinematch_positive_feedback_total` | Counter | Favorites added |
| `cinematch_drift_score` | Gauge | Per-feature drift score |
| `cinematch_latency_violations_total` | Counter | >200ms SLO violations |

### Grafana Dashboard
24 panels including request rate, latency percentiles, recommendation status, TMDB calls, CPU, memory, network, and drift metrics.

### Drift Detection
- Runs every 10 recommendation requests automatically
- Compares current query patterns against training baseline
- Thresholds: 20% mean drift, 50% variance drift
- Auto-triggers Airflow retraining DAG when drift detected

---

##  Data Versioning

```bash
# Switch between dataset versions
git checkout data-v1 -- data/tmdb_5000_movies.csv.dvc
dvc pull data/tmdb_5000_movies.csv --force
# → 4803 rows

git checkout data-v2 -- data/tmdb_5000_movies.csv.dvc
dvc pull data/tmdb_5000_movies.csv --force
# → 4800 rows
```

| Version | Tag | Rows |
|---------|-----|------|
| v1 | data-v1 | 4803 |
| v2 | data-v2 | 4800 |
| v3 | data-v3 | 4700 |

---

##  CI/CD

GitHub Actions pipeline runs on every push to `main`:

1. **Run Tests** — validates Python imports, DVC pipeline YAML
2. **Validate Config Files** — checks prometheus.yml, alerts.yml, dashboard JSON
3. **Build Docker Images** — builds backend and frontend images

View pipeline: https://github.com/Fuzail1929/DA5402-Mlops_Moviess-Recommendation-System/actions

---

##  Documentation

| Document | Description |
|----------|-------------|
| `HLD_Document.docx` | High-Level Design — architecture, components, data flow |
| `LLD_Document.docx` | Low-Level Design — API specs, module design, data models |
| `User_Manual.docx` | Non-technical user guide |
| `Test_Plan_Report.docx` | Test plan, 25 test cases, test report |
| `architecture_diagram.svg` | System architecture diagram |
| `MLproject` | MLflow project definition |

---

## 👤 Author

**Mohammed Fuzail**
DA5402 — MLOps Course Project
April 2026