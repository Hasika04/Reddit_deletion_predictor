from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def get_server():
    return {"message":"Server Live!"}

@app.get("/name")
def get_name():
    return {"name":"fastapi"}