from fastapi import FastAPI
import joblib
from pydantic import BaseModel

app = FastAPI()
model = joblib.load("model.joblib")

class IrisInput(BaseModel):
    data: list[float]

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(input_data: IrisInput):
    prediction = model.predict([input_data.data])
    return {"prediction": int(prediction[0])}