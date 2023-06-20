from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def welcome():
    return {"message": "welcome to api"}


@app.get("/health")
def health():
    return {"status": "healthy"}
