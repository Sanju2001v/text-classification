# import joblib
# class ModelPredictor:
#     def __init__(self, model_path: str):
#         self.model = joblib.load(model_path)
#     def predict(self, text: str):
#         prediction = self.model.predict([text])[0]
#         probability = self.model.predict_proba([text])[0]
#         return {"prediction": prediction, "probability": probability.tolist()}

import joblib
import numpy as np

class ModelPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, text: str):
        prediction = self.model.predict([text])[0]  # Prediction result
        probability = self.model.predict_proba([text])[0]  # Probability array

        # Convert numpy types to Python types
        prediction = int(prediction)  # Convert numpy.int64 to int
        probability = probability.tolist()  # Convert numpy array to list

        return {"prediction": prediction, "probability": probability}
