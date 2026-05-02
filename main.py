import os
import json
import base64
import re

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ================= INIT =================

app = FastAPI(title="Murmatika Vision API 🐾")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================= MEMORY =================

state_memory = {}

# ================= MODELS =================

class ChatMessageRequest(BaseModel):
    name: str
    grade: int
    message: str
    user_id: str

# ================= UTILS =================

def extract_number(text: str):
    nums = re.findall(r'\d+', text)
    return int(nums[-1]) if nums else None

def is_simple_addition(text: str):
    return re.match(r'^\d+\s*\+\s*\d+$', text.strip())

# ================= STEP ENGINE =================

def build_addition_steps(text):
    a, b = map(int, re.findall(r'\d+', text))
    to10 = 10 - a

    if b > to10:
        rest = b - to10
        return [
            {"q": f"Сколько нужно добавить к {a}, чтобы получилось 10?", "a": to10},
            {"q": f"Сколько останется от {b}, если взять {to10}?", "a": rest},
            {"q": f"Сколько будет 10 + {rest}?", "a": 10 + rest},
        ]

    return [
        {"q": f"Сколько будет {a} + {b}?", "a": a + b}
    ]

# ================= VISION =================

@app.post("/api/vision")
async def vision_ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # защита от огромных файлов
        if len(contents) > 5_000_000:
            return {"tasks": ["Фото слишком большое 📷"]}

        base64_image = base64.b64encode(contents).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
Ты анализируешь фото страницы учебника (1–5 класс).

Твоя задача:
- Найти ВСЕ задачи на странице
- Разделить их
- Исправить ошибки распознавания
- НЕ решать задачи

Верни строго JSON:

{
  "tasks": [
    "9 + 17",
    "5 + 4",
    "Ваня шел 12 минут..."
  ]
}
"""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Найди задачи на изображении"},
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

        data = json.loads(response.choices[0].message.content)

        return {
            "tasks": data.get("tasks", [])
        }

    except Exception as e:
        return {
            "tasks": [],
            "error": str(e)
        }

# ================= CHAT =================

@app.post("/api/chat/message")
def chat(data: ChatMessageRequest):
    user_id = data.user_id
    text = data.message.strip()

    # --- новая задача ---
    if user_id not in state_memory:

        if is_simple_addition(text):
            steps = build_addition_steps(text)
        else:
            return {
                "reply": "Я пока помогаю с простыми примерами 😊",
                "question": "",
                "hint": "",
                "emotion": "thinking"
            }

        state_memory[user_id] = {
            "steps": steps,
            "step": 0
        }

        return {
            "reply": "Давай решим вместе 🐾",
            "question": steps[0]["q"],
            "hint": "",
            "emotion": "thinking"
        }

    state = state_memory[user_id]
    step = state["steps"][state["step"]]

    user_num = extract_number(text)

    if user_num is None:
        return {
            "reply": "Напиши число 😊",
            "question": step["q"],
            "hint": "",
            "emotion": "thinking"
        }

    # --- правильно ---
    if user_num == step["a"]:
        state["step"] += 1

        if state["step"] >= len(state["steps"]):
            state_memory.pop(user_id)

            return {
                "reply": "Отлично! 🎉",
                "question": "",
                "hint": "",
                "emotion": "proud"
            }

        next_step = state["steps"][state["step"]]

        return {
            "reply": "Верно 👍",
            "question": next_step["q"],
            "hint": "",
            "emotion": "happy"
        }

    # --- ошибка ---
    return {
        "reply": "Попробуй ещё 🐾",
        "question": step["q"],
        "hint": "",
        "emotion": "thinking"
    }

# ================= SOLUTION =================

@app.get("/api/solution")
def solution():
    return {
        "solution": "Скоро будет подробное решение 🐾"
    }

# ================= HEALTH =================

@app.get("/health")
def health():
    return {"status": "ok"}