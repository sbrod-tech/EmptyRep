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

# ===== MEMORY =====
chat_memory = {}

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

ОТВЕЧАЙ СТРОГО В JSON:
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
    user_id: str

# ===== ROUTES =====
@app.get("/")
def root():
    return {"message": "Murmatika backend is running 🐾"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ===== TEXT CHAT (С ПАМЯТЬЮ) =====
@app.post("/api/chat/message")
def chat_message(data: ChatMessageRequest):
    try:
        user_id = data.user_id

        # создаём память если нет
        if user_id not in chat_memory:
            chat_memory[user_id] = []

        # добавляем сообщение пользователя
        chat_memory[user_id].append({
            "role": "user",
            "content": data.message
        })

        # ограничиваем историю (последние 10 сообщений)
        history = chat_memory[user_id][-10:]

        messages = [
            {"role": "system", "content": get_system_prompt(data.grade)}
        ] + history

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=messages
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        # сохраняем ответ Мурчика
        chat_memory[user_id].append({
            "role": "assistant",
            "content": content
        })

        # защита от переполнения памяти
        if len(chat_memory[user_id]) > 20:
            chat_memory[user_id] = chat_memory[user_id][-20:]

        return parsed

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
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    contents = await image.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    try:
        history = chat_memory.get(user_id, [])[-6:]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": get_system_prompt(grade)},
                *history,
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

        content = response.choices[0].message.content
        parsed = json.loads(content)

        return parsed

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
    user_id: str = Form(...),
    user_answer: str = Form(...)
):
    try:
        history = chat_memory.get(user_id, [])[-6:]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": get_system_prompt(grade)},
                *history,
                {"role": "user", "content": f"Ответ ученика: {user_answer}"}
            ]
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        return parsed

    except Exception as e:
        return {
            "reply": "Давай попробуем ещё 🐾",
            "question": "Подумай внимательно",
            "hint": str(e),
            "emotion": "thinking"
        }