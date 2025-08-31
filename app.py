from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
import xgboost as xgb

app=FastAPI()
class Prediction_input(BaseModel):
    vendor_id: float
    passenger_count: float
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: float
    pickup_weekday: float
    pickup_hour_weekofyear: float
    pickup_hour: float
    pickup_minute: float
    pickup_dt: float
    pickup_week_hour: float
    pickup_pca0: float
    pickup_pca1: float
    dropoff_pca0: float
    dropoff_pca1: float
    distance_haversine: float
    distance_dummy_manhattan: float
    direction: float
    pca_manhattan: float
    center_latitude: float
    center_longitude: float
    pickup_cluster: float
    dropoff_cluster: float


model_path="models/model.joblib"
model = load(model_path)

@app.get("/")
def home():
    return "working fine"

@app.post("/predict")
def predict(input_data:Prediction_input):
    features=[input_data.vendor_id,
    input_data.passenger_count,
    input_data.pickup_longitude,
    input_data.pickup_latitude,
    input_data.dropoff_longitude,
    input_data.dropoff_latitude,
    input_data.store_and_fwd_flag,
    input_data.pickup_weekday,
    input_data.pickup_hour_weekofyear,
    input_data.pickup_hour,
    input_data.pickup_minute,
    input_data.pickup_dt,
    input_data.pickup_week_hour,
    input_data.pickup_pca0,
    input_data.pickup_pca1,
    input_data.dropoff_pca0,
    input_data.dropoff_pca1,
    input_data.distance_haversine,
    input_data.distance_dummy_manhattan,
    input_data.direction,
    input_data.pca_manhattan,
    input_data.center_latitude,
    input_data.center_longitude,
    input_data.pickup_cluster,
    input_data.dropoff_cluster]

    transformed_feature=xgb.DMatrix([features])


    prediction=model.predict([transformed_feature])[0].item()
    return {"prediction": prediction}


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=5000)