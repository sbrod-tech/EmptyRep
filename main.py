import base64
import os
import json
import re
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

# ===== UTILS =====

def safe_eval(expr: str):
    try:
        expr = expr.replace(" ", "")
        if not re.match(r'^[0-9+\-*/().]+$', expr):
            return None
        return eval(expr)
    except:
        return None

def extract_number(text: str):
    nums = re.findall(r'\d+', text)
    return int(nums[-1]) if nums else None

# ===== ANALYZE PROBLEM =====

def analyze_problem(text, grade):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": f"""
Ты анализируешь задачу для {grade} класса.

Верни JSON:

{{
  "type": "time | arithmetic | unknown",
  "goal": "что нужно найти",
  "steps": ["шаг 1", "шаг 2"]
}}

НЕ решай задачу.
ОТВЕТ строго JSON.
"""
                },
                {"role": "user", "content": text}
            ]
        )

        return json.loads(response.choices[0].message.content)
    except:
        return {"type": "unknown"}

# ===== SYSTEM PROMPT =====

def get_system_prompt(grade):
    return f"""
Ты — Мурчик 🐾, наставник по математике для {grade} класса.

ВАЖНО:
Ты должен отвечать только в JSON формате (json_object).

ПРАВИЛА:
- НЕ давай ответ сразу
- веди по шагам
- проверяй ответы ученика
- если ошибка — мягко исправь
- не дублируй вопросы

ФОРМАТ:
{{
  "reply": "...",
  "question": "...",
  "hint": "...",
  "emotion": "happy | thinking | confused | proud | playful"
}}
"""

# ===== MODEL =====

class ChatMessageRequest(BaseModel):
    name: str
    grade: int
    message: str
    user_id: str

# ===== ROUTES =====

@app.get("/")
def root():
    return {"message": "Murmatika API running 🐾"}

@app.get("/health")
def health():
    return {"status": "ok"}

# ===== MAIN CHAT =====

@app.post("/api/chat/message")
def chat_message(data: ChatMessageRequest):
    try:
        user_id = data.user_id

        # создаём память
        if user_id not in chat_memory:
            chat_memory[user_id] = []

        # добавляем сообщение
        chat_memory[user_id].append({
            "role": "user",
            "content": data.message
        })

        # ===== КОНТРОЛЬ ОТВЕТОВ =====
        correct = safe_eval(data.message)
        user_answer = extract_number(data.message)

        if correct is not None and user_answer is not None:
            if user_answer == correct:
                return {
                    "reply": "Отлично! Это правильный ответ 🎉",
                    "question": "Хочешь решить ещё задачу?",
                    "hint": "",
                    "emotion": "proud"
                }
            else:
                return {
                    "reply": "Давай проверим ещё раз 🐾",
                    "question": f"Сколько будет {data.message}?",
                    "hint": "Попробуй сложить десятки и единицы отдельно",
                    "emotion": "thinking"
                }

        # ===== АНАЛИЗ ЗАДАЧИ =====
        analysis = analyze_problem(data.message, data.grade)

        history = chat_memory[user_id][-6:]

        messages = [
            {"role": "system", "content": get_system_prompt(data.grade)},
            {"role": "system", "content": f"Анализ задачи: {analysis}"},
            *history
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=messages
        )

        content = response.choices[0].message.content
        parsed = json.loads(content)

        # сохраняем ответ
        chat_memory[user_id].append({
            "role": "assistant",
            "content": content
        })

        # ограничиваем память
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
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": get_system_prompt(grade)},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Реши задачу"},
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
            "question": "Попробуй ещё раз?",
            "hint": str(e),
            "emotion": "confused"
        }