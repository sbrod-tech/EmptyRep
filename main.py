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

    # OCR мусор
    text = text.replace('•', ' ')
    text = text.replace('·', ' ')
    text = text.replace('|', ' ')
    text = text.replace('—', '-')

    return text.strip()

# =========================
# ✂️ SMART SPLIT
# =========================
def split_tasks_smart(text: str):
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    tasks = []
    current_task = ""

    starters = [
        "найди", "определи", "запиши", "реши",
        "вычисли", "сколько", "с какой",
        "вырежи", "начерти"
    ]

    def is_new_task(line):
        if re.match(r'^\d{2,3}[\.\)]?\s', line):
            return True
        lower = line.lower()
        return any(lower.startswith(w) for w in starters)

    for line in lines:
        if is_new_task(line):
            if current_task:
                tasks.append(current_task.strip())
            current_task = line
        else:
            if current_task:
                current_task += " " + line

    if current_task:
        tasks.append(current_task.strip())

    # фильтры
    tasks = [t for t in tasks if len(t) > 40]
    tasks = [t for t in tasks if re.search(r'[а-яА-Я]', t)]

    return tasks

# =========================
# 🔢 АНТИ-ФАНТАЗИЯ
# =========================
def contains_new_numbers(task, steps):
    task_numbers = set(re.findall(r'\d+', task))
    steps_numbers = set(re.findall(r'\d+', json.dumps(steps)))
    return not steps_numbers.issubset(task_numbers)

# =========================
# ❌ ПЛОХИЕ ШАГИ
# =========================
def is_bad_step(step):
    q = step["question"].lower()

    bad_patterns = [
        "что такое",
        "как думаешь",
        "попробуй подумать",
        "например"
    ]

    return any(p in q for p in bad_patterns)

# =========================
# 🧠 ПРОВЕРКА ЛОГИКИ
# =========================
def validate_steps(task: str, steps: list):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "user",
                    "content": f"""
Ты проверяешь шаги решения задачи.

ЗАДАЧА:
{task}

ШАГИ:
{json.dumps(steps, ensure_ascii=False)}

Проверь:
- логика последовательная?
- нет ли скачков?
- каждый шаг — действие?

Ответ:

{{ "valid": true }}
или
{{ "valid": false, "reason": "коротко" }}
"""
                }
            ],
        )

        raw = response.choices[0].message.content
        return json.loads(raw)

    except Exception:
        return {"valid": True}

# =========================
# 🧠 ГЕНЕРАЦИЯ ШАГОВ (РЕПЕТИТОР)
# =========================
def generate_steps(task: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": f"""
Ты лучший репетитор по математике для детей 1–5 класса.

=========================
ПРАВИЛА
=========================

1. Каждый шаг = КОНКРЕТНОЕ действие
2. Вопрос = простой и понятный
3. НЕЛЬЗЯ:
   - "что такое..."
   - "как думаешь"
   - "например"
4. НЕ менять числа
5. НЕ придумывать данные

=========================
ФОРМАТ
=========================

{{
  "steps": [
    {{
      "question": "...",
      "answer": "...",
      "hint": "..."
    }}
  ]
}}

=========================
ЗАДАЧА
=========================

{task}
"""
            }
        ],
    )

    raw = response.choices[0].message.content
    return json.loads(raw)

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
                            "text": "Распознай текст полностью, сохрани строки"
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
# 🚀 GENERATE API
# =========================
@app.post("/api/generate")
async def generate(req: TaskRequest):
    try:
        data = generate_steps(req.task)

        # защита от новых чисел
        if contains_new_numbers(req.task, data):
            data = generate_steps(
                req.task + "\n\nНЕ ДОБАВЛЯЙ НОВЫЕ ЧИСЛА"
            )

        # защита от плохих вопросов
        if any(is_bad_step(s) for s in data["steps"]):
            data = generate_steps(
                req.task + "\n\nНЕ задавай абстрактные вопросы"
            )

        # проверка логики
        validation = validate_steps(req.task, data["steps"])

        if not validation.get("valid", True):
            data = generate_steps(
                req.task + "\n\nИсправь логику шагов"
            )

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