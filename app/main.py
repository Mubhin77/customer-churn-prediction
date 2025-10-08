from fastapi import FastAPI, Request, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pandas as pd
import joblib
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR))

# Load model and threshold
deployment_info = joblib.load("../model/churn_model_deployed.pkl")
model = deployment_info["model"]
threshold = deployment_info["threshold"]

def predict_churn_with_threshold(model, X_input, threshold=0.4):
    proba = model.predict_proba(X_input)[:, 1]
    prediction = (proba >= threshold).astype(int)
    risk_level = [
        "High Risk" if p >= 0.7 else "Medium Risk" if p >= threshold else "Low Risk"
        for p in proba
    ]
    return prediction, proba, risk_level

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_csv")
async def upload_csv(file: UploadFile):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        preds, probs, risks = predict_churn_with_threshold(model, df, threshold)
        df["Churn Prediction"] = ["Yes" if p == 1 else "No" for p in preds]
        df["Churn Probability (%)"] = (probs * 100).round(2)
        df["Risk Level"] = risks

        csv_path = "predicted_churn.csv"
        df.to_csv(csv_path, index=False)

        return {"message": "Prediction complete", "download_link": "/download_predictions"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/download_predictions")
async def download_predictions():
    return FileResponse(path="predicted_churn.csv", filename="predicted_churn.csv", media_type="text/csv")

@app.post("/predict")
async def predict_from_form(
    request: Request,
    gender: str = Form(...),
    SeniorCitizen: int = Form(...),
    Partner: str = Form(...),
    Dependents: str = Form(...),
    tenure: int = Form(...),
    PhoneService: str = Form(...),
    MultipleLines: str = Form(...),
    InternetService: str = Form(...),
    OnlineSecurity: str = Form(...),
    OnlineBackup: str = Form(...),
    DeviceProtection: str = Form(...),
    TechSupport: str = Form(...),
    StreamingTV: str = Form(...),
    StreamingMovies: str = Form(...),
    Contract: str = Form(...),
    PaperlessBilling: str = Form(...),
    PaymentMethod: str = Form(...),
    MonthlyCharges: float = Form(...),
    TotalCharges: float = Form(...),
):
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    pred, proba, risk = predict_churn_with_threshold(model, input_data, threshold)
    result = {
        "probability": round(float(proba[0]) * 100, 2),
        "risk": risk[0],
        "churn_pred": "Yes" if pred[0] == 1 else "No"
    }
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
