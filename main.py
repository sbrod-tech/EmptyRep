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
# 🧹 ОЧИСТКА ТЕКСТА
# =========================
def clean_text(text: str):
    text = text.replace('\r', '\n')
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\.{3,}', ' ', text)  # убираем "..."
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =========================
# ✂️ УМНЫЙ SPLIT
# =========================
def split_tasks_smart(text: str):
    matches = list(re.finditer(r'(?<!\d)(\d{2,3})[\.\)]?\s', text))

    if not matches:
        return [text]

    tasks = []

    for i in range(len(matches)):
        start = matches[i].start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(text)

        task = text[start:end].strip()

        if len(task) > 20:
            tasks.append(task)

    return tasks


# =========================
# 📸 OCR (УЛУЧШЕННЫЙ)
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
Распознай ВЕСЬ текст на изображении.

ПРАВИЛА:
- НЕ сокращай
- НЕ пересказывай
- НЕ убирай слова
- СОХРАНИ порядок
- СОХРАНИ числа и единицы

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

    except Exception:
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
Ты помощник для детей (1–5 класс).

Разбей задачу на шаги.

ПРАВИЛА:
- НЕ давай сразу ответ
- веди ребёнка через вопросы
- шаги простые
- без сложных формулировок

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