import base64
import os
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ===== INIT =====
app = FastAPI(title="Murmatika API 🐾")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== SYSTEM PROMPT =====
def get_system_prompt(grade):
    return f"""
Ты — Мурчик 🐾, добрый кот-наставник по математике для {grade} класса.

ПРАВИЛА:
- НЕ давай готовый ответ
- задавай вопросы
- помогай думать
- говори просто
- будь дружелюбным
- максимум 2-3 предложения

ФОРМАТ JSON:
{{
  "reply": "...",
  "question": "...",
  "hint": "...",
  "emotion": "happy | thinking | confused | proud | playful"
}}
"""

# ===== MODELS =====
class ChatMessageRequest(BaseModel):
    name: str
    grade: int
    message: str

# ===== ROUTES =====
@app.get("/")
def root():
    return {"message": "Murmatika backend is running 🐾"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ===== TEXT CHAT =====
@app.post("/api/chat/message")
def chat_message(data: ChatMessageRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": get_system_prompt(data.grade)},
                {"role": "user", "content": data.message}
            ]
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "reply": "Хмм… давай попробуем ещё раз 🐾",
            "question": "С чего можно начать?",
            "hint": str(e),
            "emotion": "thinking"
        }

# ===== IMAGE CHAT =====
@app.post("/api/chat/image")
async def chat_image(
    name: str = Form(...),
    grade: int = Form(...),
    image: UploadFile = File(...)
):
    contents = await image.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": get_system_prompt(grade)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Реши задачу с фото"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "reply": "Мурчик не понял картинку 😿",
            "question": "Попробуй сфотографировать ещё раз?",
            "hint": str(e),
            "emotion": "confused"
        }

# ===== ANSWER CHECK =====
@app.post("/api/chat/answer")
async def chat_answer(
    name: str = Form(...),
    grade: int = Form(...),
    user_answer: str = Form(...)
):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": get_system_prompt(grade)},
                {"role": "user", "content": f"Ответ ученика: {user_answer}"}
            ]
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "reply": "Давай попробуем ещё 🐾",
            "question": "Подумай внимательно",
            "hint": str(e),
            "emotion": "thinking"
        }