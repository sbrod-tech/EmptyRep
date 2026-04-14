from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Math tutor backend works"}

@app.get("/health")
def health():
    return {"status": "ok"}