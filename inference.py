import pickle
import numpy as np

# Load the ML model from pkl file
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Make prediction
def make_prediction(model, input_data):
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    # Load model
    model = load_model('model.pkl')
    
    # Example input data
    input_data = np.array([[1, 2, 3, 4, 5]])
    
    # Make prediction
    result = make_prediction(model, input_data)
    print(f"Prediction: {result}")
