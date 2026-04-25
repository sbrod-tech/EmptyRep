from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Math Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Math Tutor Backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


class ChatMessageRequest(BaseModel):
    name: str
    grade: int
    message: str


@app.post("/api/chat/message")
def chat_message(data: ChatMessageRequest):
    return {
        "reply": f"{data.name}, отлично! Давай решим вместе.",
        "question": "Как ты думаешь, с чего лучше начать?",
        "hint": "Попробуй сначала найти десятки, а потом единицы.",
        "emotion": "thinking"
    }


@app.post("/api/chat/image")
async def chat_image(
    name: str = Form(...),
    grade: int = Form(...),
    image: UploadFile = File(...)
):
    return {
        "reply": f"{name}, я получил фото задачи!",
        "question": "Давай подумаем: что в задаче нужно найти?",
        "hint": "Сначала внимательно посмотри на числа и знак действия.",
        "emotion": "encourage",
        "file_name": image.filename,
        "content_type": image.content_type
    }