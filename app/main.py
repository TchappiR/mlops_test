from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model.joblib") # Charge le modèle sauvegardé [cite: 25]

class IrisInput(BaseModel):
    data: list[float]

@app.get("/health") # Endpoint de santé [cite: 26, 29]
def health():
    return {"status": "healthy"}

@app.post("/predict") # Endpoint de prédiction [cite: 27, 30, 31]
def predict(input_data: IrisInput):
    prediction = model.predict([input_data.data])
    return {"prediction": int(prediction[0])}