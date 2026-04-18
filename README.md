# Student Performance Prediction – MLOps + DevOps

This project predicts the final grade of a student based on internal assessment scores, attendance and related features, using a regression model exposed via a FastAPI web API. The focus is on end-to-end MLOps and DevOps: data processing, experiment tracking with MLflow, containerization with Docker, CI/CD with GitHub Actions, and deployment on Render. [web:155][web:156]

## Tech Stack

- Python, scikit-learn, pandas
- FastAPI for serving the model [web:169]
- MLflow for experiment tracking [web:173]
- Docker for containerization
- GitHub Actions for CI/CD
- Render for cloud deployment [web:171][web:174]

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/student-mlops.git
cd student-mlops
```

### 2. Create and activate virtual environment (optional)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model and save `model.joblib`

```bash
python src/models/train.py
```

This trains the regression model on the student dataset and saves the trained model to `models/model.joblib`.

### 5. Start the FastAPI app

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Then open:

- API docs (Swagger UI): http://127.0.0.1:8000/docs [web:169]
- Health check: http://127.0.0.1:8000/health

## MLOps Overview 

### Data Processing 

- The dataset contains student features such as attendance, internal assessment scores and previous performance.
- Data is loaded and processed in the data script (for example `src/data/make_dataset.py`), including feature selection and train/test split.
- Numeric features are cleaned / transformed as needed before model training. [web:155][web:156]

### Model Building

- A regression model (e.g. Linear Regression / ElasticNet) is implemented in `src/models/train.py`.
- The script trains the model on the training split and evaluates on the test set.
- The trained model is serialized and saved as `models/model.joblib` for reuse in the API. [web:156][web:161]

### Experiment Tracking with MLflow 

- MLflow Tracking is used to log experiments: hyperparameters, metrics and artifacts.
- Each run logs parameters (such as regularization strength), metrics (RMSE, MAE, R²) and model artifacts. [web:170][web:173]
- The MLflow UI can be launched with:

```bash
mlflow ui
```

and opened in the browser at http://127.0.0.1:5000 for comparing runs.

### Model Deployment 

- The trained model is loaded in a FastAPI application in `src/api/main.py`.
- The `/predict` endpoint accepts a JSON payload with student features and returns the predicted final grade.
- Input validation is handled with Pydantic models, and the app is served via Uvicorn. [web:165][web:169]

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "attendance": 85,
        "ia1_marks": 18,
        "ia2_marks": 19,
        "ia3_marks": 20
      }'
```

### Performance Evaluation 

- The model is evaluated on a held-out test set using metrics such as RMSE, MAE and R². [web:161][web:164]
- These metrics are printed in the training script and logged to MLflow for each run.
- Additional plots (residuals, error distribution, feature importance) are generated and discussed in the report.

## DevOps Overview 

### Version Control with Git 

- The entire project (data processing, model code, API, Dockerfile, CI config) is version-controlled using Git.
- Development is done through incremental commits, and the repository is hosted on GitHub. [web:156][web:160]

### CI/CD Pipeline with GitHub Actions 

- A GitHub Actions workflow (for example `.github/workflows/ci.yml`) runs on every push to the main branch.
- The pipeline:
  - Installs Python dependencies.
  - Runs tests / checks.
  - Builds the Docker image to ensure the app and dependencies are valid. [web:160][web:163]
- A green check on the latest commit confirms the CI pipeline passes.

### Containerization with Docker 

- The project includes a `Dockerfile` that:
  - Uses a Python base image.
  - Installs dependencies from `requirements.txt`.
  - Copies the FastAPI application and the trained `model.joblib`.
  - Exposes port `8000` and starts the app with Uvicorn. [web:160]

Build and run locally:

```bash
docker build -t student-mlops .
docker run -p 8000:8000 student-mlops
```

Then open http://127.0.0.1:8000/docs. [web:171][web:174]

### Cloud Deployment on Render 

- The Docker image is deployed as a web service on Render. [web:171][web:174]
- Render automatically builds the image from the GitHub repository and runs the container.
- The live service is available at:

```text
https://student-mlops-docker.onrender.com
```

- Health check endpoint (GET):

```text
https://student-mlops-docker.onrender.com/health
```

- API docs (Swagger UI):

```text
https://student-mlops-docker.onrender.com/docs
```
## 📸 Screenshots

### 🔬 MLflow Experiment Tracking
![MLflow Runs](outputs/mlflow-runs.png)

---

### 📊 Model Performance (Feature Importance)
![Feature Importance](outputs/feature-imp.png)

---

### ⚙️ CI/CD Pipeline (GitHub Actions)
![CI/CD Pipeline](outputs/ci-cd.png)

---


### 🌐 Deployment – Live Service
![Render Live](outputs/render-log.png)

---

### ❤️ API Health Check
![Health Endpoint](outputs/deployment-health.png)

---

### 🤖 Prediction Endpoint
![Predict Endpoint](outputs/deployment-predict.png)
- This `README.md` documents:
  - Project objective and dataset.
  - How to run the project locally.
  - How experiments are tracked with MLflow.
  - How CI/CD, Docker and Render deployment are configured. [web:165][web:171]
- Additional screenshots (MLflow UI, CI pipeline, Docker build, and Render dashboard) are included in the separate project report for evaluation.

## High-level Architecture

The project follows an end-to-end MLOps and DevOps workflow. The data is processed and used to train a regression model, whose experiments are tracked using MLflow. The best model is serialized as `model.joblib` and loaded inside a FastAPI application that exposes `/health` and `/predict` endpoints. The application is containerized with Docker, tested and built in a GitHub Actions CI/CD pipeline, and deployed as a Docker-based web service on Render, where the public URL can be used to obtain predictions in real time. [web:155][web:160][web:171]
