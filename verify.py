import pickle

class ModelPredictor:
    def __init__(self, model_path):
        # Load the pre-trained model from the file
        with open(model_path, 'rb') as file:
            self.model = pickle.load(file)

    def predict(self, text):
        # Preprocess the text if needed
        # For example, tokenization, vectorization, etc.
        # Assuming the model expects a single string input
        if isinstance(text, str):
            # Model.predict() assumes text is preprocessed and vectorized
            return self.model.predict([text])[0]
        else:
            raise ValueError("Input must be a string")

# Path to the model file
model_path = "model/svm_model.pkl"

# Initialize the predictor
predictor = ModelPredictor(model_path)

# Test the model with some example inputs
test_inputs = [
    "This is a positive sentiment example.",
    "This is a negative sentiment example.",
    "Neutral sentiment is also possible."
]

print("Testing the model predictions:")
for i, text in enumerate(test_inputs, 1):
    try:
        prediction = predictor.predict(text)
        print(f"Test {i}: Input: '{text}' | Prediction: {prediction}")
    except Exception as e:
        print(f"Test {i}: Input: '{text}' | Error: {str(e)}")
