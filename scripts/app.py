from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Define the expected input columns
columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Define the result mapping
dict_res = {0: "Not-Diabetes", 1: "Diabetes"}

# Load the pre-trained pipeline
pipeline_path = "../models/pipeline.pkl"
with open(pipeline_path, "rb") as pipeline_file:
    pipeline = pickle.load(pipeline_file)

# Define the input data model
class DataInput(BaseModel):
    data: list

# Define the prediction endpoint
@app.post("/predict")
async def predict(input_data: DataInput):
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(input_data.data, columns=columns)
        
        # Make predictions using the pipeline
        predictions = pipeline.predict(df)
        
        # Map the predictions to human-readable labels
        results = [dict_res[pred] for pred in predictions]

        return {"predictions": results}
        
    except Exception as e:
        print("Error: ", str(e))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
