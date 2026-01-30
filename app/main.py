import pickle
import os
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model.pkl")
tfidf_path = os.path.join(BASE_DIR, "vectorizer.pkl")

with open("C:/Users/LENOVO/Downloads/app/vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("C:/Users/LENOVO/Downloads/app/model.pkl", "rb") as f:
    model = pickle.load(f)

class Post(BaseModel):
    title: str

@app.post("/predict")
def predict(post: Post):
    text_vector = tfidf.transform([post.title])
    prediction = model.predict(text_vector)
    return {"prediction": int(prediction[0])}

@app.get("/")
def read_root():
    return {"message": "My Reddit App is running!"}