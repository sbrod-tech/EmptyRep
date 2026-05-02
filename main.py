from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import json
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
# 📸 OCR (Vision)
# =========================
@app.post("/api/vision")
async def vision(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Распознай текст задачи"},
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

        text = response.choices[0].message.content

        return {"text": text}

    except Exception as e:
        return {"text": ""}


# =========================
# ✂️ SPLIT задач
# =========================
@app.post("/api/split")
async def split(req: TextRequest):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": f"""
Раздели текст на отдельные задачи.

Верни JSON:
{{ "tasks": ["...", "..."] }}

Текст:
{req.text}
""",
                }
            ],
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        return {"tasks": data.get("tasks", [])}

    except Exception as e:
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

ВАЖНО:
- НЕ давай ответ сразу
- веди через вопросы
- каждый шаг:
  - question (вопрос)
  - answer (правильный ответ)
  - hint (подсказка)

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
""",
                }
            ],
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        return data

    except Exception as e:
        return {
            "steps": [
                {
                    "question": "Не удалось разобрать задачу 😢",
                    "answer": "",
                    "hint": "Попробуй перефотографировать"
                }
            ]
        }