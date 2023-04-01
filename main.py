from fastapi import FastAPI, Request
import pickle
import numpy as np

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Congratz! You've made it here!!!"}


@app.post("/predict")
async def predict(request: Request):
    obj = await request.json()

    try:
        fields = str(obj["data"])
        if fields is not None and len(fields)!=0:

            result = [str(obj["data"])]
            model_path = "ML_API_Pipeline.pickle"
            classifier = pickle.load(open(model_path, "rb"))
            prediction = str(classifier.predict(result)[0])
            conf_score = np.max(classifier.predict_proba(result))*100

            conf_score = f"{conf_score:.2f}%"
            predictions = {
                "error": "0",
                "message": "Successful",
                "prediction": prediction,
                "confidence_score": conf_score
            }

        else:
            predictions = {
                "error": "1",
                "message": "Invalid Parameters"
            }
        

    except Exception as e:
        predictions = {
            "error": "2",
            "message": str(e)
        }

    
    return (predictions)