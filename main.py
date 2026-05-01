import os
import json
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ================= INIT =================

app = FastAPI(title="Murmatika API 🐾")

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

def is_new_problem(text: str):
    t = text.lower()
    return (
        len(text) > 20 and
        any(w in t for w in [
            "сколько", "во сколько", "найди",
            "пошел", "встретились", "скорость"
        ])
    )

def parse_user_answer(text: str):
    t = text.lower().replace(",", ".").strip()

    # время
    m_time = re.search(r'(\d{1,2})[:.](\d{2})', t)
    if m_time:
        h, m = int(m_time.group(1)), int(m_time.group(2))
        return {"type": "time", "value": h * 60 + m}

    # число
    m_num = re.search(r'(\d+(\.\d+)?)', t)
    if m_num:
        val = float(m_num.group(1))
        return {"type": "number", "value": val}

    return {"type": "unknown", "value": None}

def is_close(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except:
        return False

def detect_error(user, correct):
    if user["type"] == "unknown":
        return "no_number"

    if correct["type"] == "number":
        if not is_close(user["value"], correct["value"]):
            return "wrong_value"

    if correct["type"] == "time":
        if user["value"] != correct["value"]:
            return "wrong_time"

    return "ok"

# ================= GPT STEP GENERATION =================

def generate_steps(problem, grade):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": f"""
Разбей задачу для {grade} класса на шаги.

ВАЖНО:
- каждый шаг = вопрос
- у каждого шага есть ответ

Формат:

{{
  "steps": [
    {{ "question": "...", "answer": 36 }},
    {{ "question": "...", "answer": 5 }}
  ]
}}

Ответ строго JSON
"""
                },
                {"role": "user", "content": problem}
            ]
        )

        return json.loads(response.choices[0].message.content)["steps"]

    except Exception:
        return [
            {"question": "Попробуй понять условие задачи", "answer": 0}
        ]

# ================= GPT ERROR EXPLAIN =================

def explain_error(step, user_text, correct_value, error_type, grade):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": f"""
Ты Мурчик 🐾 — учитель {grade} класса.

Шаг:
{step}

Ответ ученика:
{user_text}

Правильный ответ:
{correct_value}

Тип ошибки:
{error_type}

Сделай:
- короткое объяснение (1 строка)
- повтори вопрос
- дай подсказку

Строго JSON:
{{
  "reply": "...",
  "question": "...",
  "hint": "...",
  "emotion": "thinking"
}}
"""
                }
            ]
        )

        return json.loads(response.choices[0].message.content)

    except Exception:
        return {
            "reply": "Давай попробуем ещё раз 🐾",
            "question": step,
            "hint": "Подумай внимательно",
            "emotion": "thinking"
        }

# ================= ROUTES =================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat/message")
def chat(data: ChatMessageRequest):
    user_id = data.user_id
    text = data.message.strip()

    # ===== NEW PROBLEM =====
    if user_id not in state_memory or is_new_problem(text):

        steps = generate_steps(text, data.grade)

        state_memory[user_id] = {
            "problem": text,
            "steps": steps,
            "current_step": 0
        }

        return {
            "reply": "Давай решим задачу вместе 🐾",
            "question": steps[0]["question"],
            "hint": "",
            "emotion": "thinking"
        }

    # ===== CURRENT STATE =====
    state = state_memory[user_id]
    step_data = state["steps"][state["current_step"]]

    correct = {
        "type": "number",
        "value": step_data["answer"]
    }

    user = parse_user_answer(text)
    error_type = detect_error(user, correct)

    # ===== CORRECT =====
    if error_type == "ok":
        state["current_step"] += 1

        if state["current_step"] >= len(state["steps"]):
            return {
                "reply": "Отлично! 🎉",
                "question": "",
                "hint": "",
                "emotion": "proud"
            }

        next_step = state["steps"][state["current_step"]]

        return {
            "reply": "Верно 👍",
            "question": next_step["question"],
            "hint": "",
            "emotion": "happy"
        }

    # ===== ERROR =====
    return explain_error(
        step=step_data["question"],
        user_text=text,
        correct_value=step_data["answer"],
        error_type=error_type,
        grade=data.grade
    )