from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from datetime import timedelta
from contextlib import asynccontextmanager
import pandas as pd
import sys
import os
import shutil
import uuid
import logging
import re
from typing import List

# Ensure src is in pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.engine import metrics, insights, nlp, ingest, ml, ingest_ai
from src.api import auth
from src.config import settings

# Global State for Data
# We use a simple in-memory cache for the dataset. 
# In a production environment, this would likely be a database (PostgreSQL/Redis).
DATA_CACHE = {}
UPLOAD_DIR = settings.upload_dir
NORMALIZED_FILE = settings.normalized_file
MAX_UPLOAD_BYTES = settings.max_upload_size_mb * 1024 * 1024
ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx", ".pdf"}
SAFE_FILE_RE = re.compile(r"[^A-Za-z0-9._-]+")
logger = logging.getLogger(__name__)

# Create upload dir if not exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Executes on startup and shutdown.
    
    1. Startup: Loads the normalized CSV data into memory (DATA_CACHE) for fast access.
    2. Shutdown: Clears the cache.
    """
    try:
        # Step 1: Check for the latest uploaded and normalized dataset
        latest_file = None
        if os.path.exists(UPLOAD_DIR):
            upload_files = [os.path.join(UPLOAD_DIR, f) for f in os.listdir(UPLOAD_DIR) if f.endswith("_normalized.csv")]
            if upload_files:
                latest_file = max(upload_files, key=os.path.getmtime)
                
        # Step 2: Use the latest upload, or fall back to the default file
        csv_path = latest_file if latest_file else NORMALIZED_FILE
        
        if not os.path.exists(csv_path):
             csv_path = os.path.join("..", "..", NORMALIZED_FILE)
             
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            DATA_CACHE["df"] = df
            print(f"Loaded {len(df)} records from {csv_path}.")
        else:
            print(f"Default data {csv_path} not found. Waiting for upload.")
            DATA_CACHE["df"] = pd.DataFrame()
            
    except Exception as e:
        print(f"Error loading data: {e}")
        DATA_CACHE["df"] = pd.DataFrame()
        
    yield
    DATA_CACHE.clear()

app = FastAPI(
    title="Graide API",
    description="API for accessing student academic performance insights",
    version="1.1.0",
    lifespan=lifespan
)


def _sanitize_filename(filename: str) -> str:
    if not filename:
        return "upload"
    cleaned = SAFE_FILE_RE.sub("_", filename).strip("._")
    return cleaned or "upload"


async def _validate_upload_size(file: UploadFile) -> None:
    size = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        if size > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds max upload size of {settings.max_upload_size_mb}MB",
            )
    await file.seek(0)

# CORS (Allow frontend to connect)
# allow_credentials must be False if allow_origins contains "*"
_allow_credentials = "*" not in settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "path": request.url.path,
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path, exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "path": request.url.path,
            }
        },
    )

@app.get("/")
def health_check():
    return {"status": "ok", "records_loaded": len(DATA_CACHE.get("df", []))}

def get_filtered_df(current_user: auth.UserResponse):
    df = DATA_CACHE.get("df")
    if df is None or df.empty:
         raise HTTPException(status_code=503, detail="Data not loaded")
         
    if current_user.role == "teacher":
        if not current_user.subjects:
            raise HTTPException(status_code=403, detail="Teacher has no assigned subjects")
        df = df[df["subject"].isin(current_user.subjects)]
        if df.empty:
            raise HTTPException(status_code=404, detail="No data available for assigned subjects")
    return df

# --- Auth Endpoints ---

@app.post("/api/v1/auth/login", response_model=auth.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = auth.get_user_by_email(form_data.username)
    if not user or not auth.verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    subjects_list = [s.strip() for s in user.subjects.split(",")] if user.subjects else []
    user_resp = auth.UserResponse(
        id=user.id, email=user.email, name=user.name, 
        role=user.role, subjects=subjects_list
    )
    
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "user": user_resp}

@app.get("/api/v1/auth/me", response_model=auth.UserResponse)
async def read_users_me(current_user: auth.UserResponse = Depends(auth.get_current_user)):
    return current_user

# --- Ingestion Endpoints ---

@app.post("/api/v1/datasets/upload")
async def upload_dataset(file: UploadFile = File(...), current_user: auth.UserResponse = Depends(auth.get_current_user)):
    """
    Upload a raw CSV, Excel, or PDF file. Returns a dataset_id.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can upload datasets.")
        
    await _validate_upload_size(file)
    safe_filename = _sanitize_filename(file.filename or "")
    ext = os.path.splitext(safe_filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    
    dataset_id = str(uuid.uuid4())
    file_location = os.path.join(UPLOAD_DIR, f"{dataset_id}_{safe_filename}")
    
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
        
    return {
        "message": "File uploaded successfully",
        "dataset_id": dataset_id,
        "filename": safe_filename,
        "status": "pending_processing"
    }

@app.post("/api/v1/datasets/{dataset_id}/process")
def process_dataset(dataset_id: str, background_tasks: BackgroundTasks, current_user: auth.UserResponse = Depends(auth.get_current_user)):
    """
    Trigger processing (normalization + loading) of an uploaded dataset.
    This runs the data cleaning and normalization pipeline.
    """
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can process datasets.")
    # Find the file
    files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(dataset_id)]
    if not files:
        raise HTTPException(status_code=404, detail="Dataset not found")
        
    raw_path = os.path.join(UPLOAD_DIR, files[0])
    
    try:
        # First, try the fast, local heuristic parser
        raw_df = ingest.parse_file(raw_path)
        
        # Check if local parser succeeded
        if not ingest.validate_schema(raw_df):
             raise ValueError("Insufficient headers for standard heuristic parse.")
             
        # Normalize
        normalized_df = ingest.normalize_dataset(raw_df)

    except Exception as heuristic_error:
        print(f"Heuristic parser failed or rejected schema: {heuristic_error}")
        print("Falling back to AI Data Extractor (Gemini)...")
        
        try:
            # Determine mime type for Gemini
            ext = os.path.splitext(raw_path)[1].lower()
            mime_type = "text/csv"
            if ext in [".xls", ".xlsx"]:
                mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            elif ext == ".pdf":
                mime_type = "application/pdf"
            
            # Use the LLM to extract directly to the normalized schema
            extractor = ingest_ai.LLMDataExtractor()
            normalized_df = extractor.extract_from_file(raw_path, mime_type)
            
        except Exception as ai_error:
            # If AI also fails, return the combined error
            import traceback
            err_detail = traceback.format_exc()
            raise HTTPException(status_code=500, detail=f"Both Parsers Failed.\n\nHeuristic Error: {heuristic_error}\n\nAI Error: {ai_error}\n\n{err_detail}")
            
    try:
        # Save Normalized
        processed_path = os.path.join(UPLOAD_DIR, f"{dataset_id}_normalized.csv")
        normalized_df.to_csv(processed_path, index=False)
        
        # Load into Memory (Active Dataset)
        DATA_CACHE["df"] = normalized_df
        
        return {
            "message": "Dataset processed and loaded successfully",
            "records": len(normalized_df),
            "status": "active"
        }
        
    except Exception as e:
        import traceback
        err_detail = traceback.format_exc()
        print(err_detail)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}\n\n{err_detail}")

# --- Analytics Endpoints ---

@app.get("/api/v1/students")
def list_students(limit: int = 100, search: str = ""):
    df = DATA_CACHE.get("df")
    if df is None or df.empty:
         raise HTTPException(status_code=503, detail="Data not loaded")
    
    # Get unique students
    # We assume 'student_id' exists. If 'name' existed we would return that too.
    unique_students = df[["student_id"]].drop_duplicates()
    
    if search:
        unique_students = unique_students[unique_students["student_id"].astype(str).str.contains(search)]
    
    # Pagination
    results = unique_students.head(limit).to_dict(orient="records")
    
    return {
        "count": len(unique_students),
        "results": results
    }

@app.get("/api/v1/students/{student_id}/summary")
def get_student_summary(student_id: int):
    """
    Returns a high-level summary of a student's academic history.
    Includes overall average, total semesters, and automated insights.
    """
    df = DATA_CACHE.get("df")
    if df is None or df.empty:
         raise HTTPException(status_code=503, detail="Data not loaded")

    # Filter for student
    student_df = df[df["student_id"] == student_id]
    
    if student_df.empty:
        raise HTTPException(status_code=404, detail="Student not found")

    # 1. Basic Stats
    total_avg = student_df["score"].mean()
    
    # 2. Insights
    # We need the full history averages to compute deltas
    all_avgs = metrics.calculate_student_averages(df)
    student_avgs = all_avgs[all_avgs["student_id"] == student_id]
    
    deltas = metrics.calculate_performance_deltas(student_avgs)
    raw_insights = insights.generate_student_insights(deltas)
    
    # NLP
    narrative_insights = [nlp.explain_insight(i) for i in raw_insights]
    
    return {
        "student_id": student_id,
        "overall_average": round(total_avg, 2),
        "total_semesters": student_df["semester"].nunique(),
        "insights": narrative_insights,
        "history": student_avgs.to_dict(orient="records")
    }

@app.get("/api/v1/cohort/trends")
def get_cohort_trends(current_user: auth.UserResponse = Depends(auth.get_current_user)):
    df = get_filtered_df(current_user)
         
    trends = metrics.calculate_cohort_trends(df)
    
    return {
        "trends": trends.to_dict(orient="records")
    }

@app.get("/api/v1/cohort/correlations")
def get_cohort_correlations(current_user: auth.UserResponse = Depends(auth.get_current_user)):
    """
    Returns correlation matrix data between subjects.
    Used for the heatmap visualization.
    """
    df = get_filtered_df(current_user)
    
    # 1. Calculate Matrix
    corr_matrix = metrics.calculate_subject_correlations(df)
    
    # 2. Extract Key Insights (Strong correlations)
    insights_list = insights.generate_cohort_correlations(corr_matrix)
    narrative_insights = [nlp.explain_insight(i) for i in insights_list]
    
    # 3. Format full matrix for a Heatmap implementation on frontend
    # We return it as a list of {x: subject_a, y: subject_b, value: correlation_coefficient}
    matrix_data = []
    for subj_a in corr_matrix.columns:
        for subj_b in corr_matrix.columns:
            matrix_data.append({
                "x": subj_a,
                "y": subj_b,
                "value": round(corr_matrix.loc[subj_a, subj_b], 3)
            })
            
    return {
        "insights": narrative_insights,
        "heatmap_data": matrix_data
    }

# --- Machine Learning API Endpoints ---

@app.get("/api/v1/students/{student_id}/ml/profile")
def get_student_ml_profile(student_id: int):
    """
    Returns the ML Cluster Profile for a student (e.g. 'Consistent High Performer').
    Uses K-Means clustering on the entire dataset to segment the student.
    """
    df = DATA_CACHE.get("df")
    if df is None or df.empty:
         raise HTTPException(status_code=503, detail="Data not loaded")

    # Filter for student to check existence
    if student_id not in df["student_id"].values:
        raise HTTPException(status_code=404, detail="Student not found")

    # 1. Feature Extraction (On entire dataset for context)
    # We need the whole dataset to define the clusters relative to the population.
    extractor = ml.FeatureExtractor(df)
    features_df = extractor.extract_features()
    
    # 2. Clustering (Train on fly for now - perfectly fine for <100k records)
    # In production, this model would be trained periodically and saved (pickled).
    model = ml.StudentClusterModel(n_clusters=4)
    model.train(features_df)
    
    # 3. Get Result
    result = model.get_student_cluster(student_id, features_df)
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to generate ML profile")
        
    return result

@app.get("/api/v1/students/{student_id}/ml/forecast")
def get_student_forecast(student_id: int):
    """
    Returns a performance forecast for the next semester.
    Uses Linear Regression on the student's personal history.
    """
    df = DATA_CACHE.get("df")
    if df is None or df.empty:
         raise HTTPException(status_code=503, detail="Data not loaded")

    # Filter for student to check existence
    if student_id not in df["student_id"].values:
        raise HTTPException(status_code=404, detail="Student not found")

    # Predict
    forecaster = ml.PerformanceForecaster(df)
    result = forecaster.forecast_next_semester(student_id)
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to generate forecast")
        
    return result

@app.get("/api/v1/students/{student_id}/ml/risk")
def get_student_risk(student_id: int):
    """
    Returns a risk assessment (Low, Moderate, Critical).
    Analyzes trends, drops, and variance.
    """
    df = DATA_CACHE.get("df")
    if df is None or df.empty:
         raise HTTPException(status_code=503, detail="Data not loaded")

    # Filter for student to check existence
    if student_id not in df["student_id"].values:
        raise HTTPException(status_code=404, detail="Student not found")

    # Detect Risk
    detector = ml.RiskDetector(df)
    result = detector.assess_student_risk(student_id)
    
    if not result:
        raise HTTPException(status_code=500, detail="Failed to assess risk")
        
    return result

@app.get("/api/v1/cohort/subjects/analysis")
def get_subject_analysis(current_user: auth.UserResponse = Depends(auth.get_current_user)):
    """
    Returns PCA analysis of subjects to visualize their relationships.
    """
    df = get_filtered_df(current_user)

    analyzer = ml.SubjectAnalyzer(df)
    result = analyzer.analyze_subjects()
    
    if "error" in result:
        # Not a server error, just insufficient data for this specific analysis
        return result
        
    return result
