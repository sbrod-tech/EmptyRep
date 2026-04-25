from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Math Tutor API")

# ---- проверка ----
@app.get("/health")
def health():
    return {"status": "ok"}


# ---- модель запроса ----
class ChatRequest(BaseModel):
    student_id: str
    message: str


# ---- основной endpoint ----
@app.post("/api/chat/message")
def chat_message(data: ChatRequest):
    
    # пока заглушка (потом подключим ИИ)
    return {
        "reply": "Давай подумаем вместе",
        "guiding_question": "Сколько будет 40 + 20?",
        "hint": "Сначала сложи десятки",
        "show_final_answer": False
    }