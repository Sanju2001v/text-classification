# from fastapi import APIRouter, HTTPException
# from model.predict import ModelPredictor
# from model.monitor import monitor_prediction_time
# router = APIRouter()
# predictor = ModelPredictor("model/svm_model.pkl")
# monitor = monitor_prediction_time()
# @router.post("/predict/")
# @monitor
# def predict(text: str):
#     try:
#         result = predictor.predict(text)
#         return {"status": "success", "data": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
from fastapi import APIRouter, HTTPException
from model.predict import ModelPredictor
from model.monitor import monitor_prediction_time  # Import the monitoring decorator

router = APIRouter()
predictor = ModelPredictor("model/svm_model.pkl")

@router.post("/predict/")
@monitor_prediction_time  # Apply the decorator correctly (without calling it as a function)
def predict(text: str):
    try:
        result = predictor.predict(text)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))