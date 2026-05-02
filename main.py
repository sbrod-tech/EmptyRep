from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import json
import re
from openai import OpenAI

app = FastAPI()

# =========================
# 🌍 CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🤖 OpenAI
# =========================
client = OpenAI()

# =========================
# 📦 MODELS
# =========================
class TextRequest(BaseModel):
    text: str


class TaskRequest(BaseModel):
    task: str


# =========================
# ❤️ HEALTH
# =========================
@app.get("/")
def root():
    return {"status": "ok"}


# =========================
# 🧹 CLEAN TEXT
# =========================
def clean_text(text: str):
    text = text.replace('\r', '\n')
    text = re.sub(r'\n+', '\n', text)

    # убираем мусорные символы OCR
    text = text.replace('•', ' ')
    text = text.replace('·', ' ')
    text = text.replace('|', ' ')
    text = text.replace('—', '-')

    return text.strip()


# =========================
# ✂️ УМНЫЙ SPLIT (УЛУЧШЕННЫЙ)
# =========================
def split_tasks_smart(text: str):
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    tasks = []
    current_task = ""

    for line in lines:
        # старт новой задачи
        if re.match(r'^\d{2,3}[\.\)]?\s', line):
            if current_task:
                tasks.append(current_task.strip())
            current_task = line
        else:
            if current_task:
                current_task += " " + line

    if current_task:
        tasks.append(current_task.strip())

    # =========================
    # 🧠 ФИЛЬТРЫ
    # =========================

    # убираем короткие куски
    tasks = [t for t in tasks if len(t) > 40]

    # убираем чисто числовой мусор
    tasks = [
        t for t in tasks
        if re.search(r'[а-яА-Я]', t)
    ]

    return tasks


# =========================
# 📸 OCR
# =========================
@app.post("/api/vision")
async def vision(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """
Распознай ВЕСЬ текст на изображении максимально точно.

ПРАВИЛА:
- НЕ сокращай текст
- НЕ пересказывай
- СОХРАНИ порядок строк
- СОХРАНИ переносы строк
- НЕ объединяй абзацы
- НЕ исправляй смысл

Верни только текст.
"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
        )

        text = response.choices[0].message.content or ""

        return {"text": text}

    except Exception as e:
        return {"text": ""}


# =========================
# ✂️ SPLIT API
# =========================
@app.post("/api/split")
async def split(req: TextRequest):
    try:
        text = clean_text(req.text)
        tasks = split_tasks_smart(text)

        return {"tasks": tasks}

    except Exception:
        return {"tasks": [req.text]}


# =========================
# 🧠 GENERATE STEPS
# =========================
@app.post("/api/generate")
async def generate(req: TaskRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": f"""
Ты помощник по математике для детей 1–5 класса.

Разбей задачу на маленькие шаги.

ПРАВИЛА:
- НЕ давай сразу ответ
- веди через вопросы
- шаги простые и понятные
- дружелюбный стиль

Верни JSON:

{{
  "steps": [
    {{
      "question": "...",
      "answer": "...",
      "hint": "..."
    }}
  ]
}}

Задача:
{req.task}
"""
                }
            ],
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        return data

    except Exception:
        return {
            "steps": [
                {
                    "question": "Не удалось разобрать задачу 😢",
                    "answer": "",
                    "hint": "Попробуй ещё раз"
                }
            ]
        }