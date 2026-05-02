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

def extract_number(text: str):
    nums = re.findall(r'\d+', text)
    return int(nums[-1]) if nums else None

# ================= STEP GENERATION =================

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
- у каждого шага есть числовой ответ

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
        return [{"question": "Попробуй разобраться в задаче", "answer": 0}]

# ================= ERROR EXPLAIN =================

def explain_error(step, user_text, correct_value, grade):
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

Сделай:
- короткое объяснение (1 строка)
- повтори вопрос
- 1 подсказку

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
            "current_step": 0,
            "last_question": None
        }

        first_question = steps[0]["question"]

        state_memory[user_id]["last_question"] = first_question

        return {
            "reply": "Давай решим задачу вместе 🐾",
            "question": first_question,
            "hint": "",
            "emotion": "thinking"
        }

    # ===== CURRENT STATE =====
    state = state_memory[user_id]
    step_data = state["steps"][state["current_step"]]

    correct = step_data["answer"]
    user_number = extract_number(text)

    # ===== НЕ ПОНЯЛ ОТВЕТ =====
    if user_number is None:
        return {
            "reply": "Попробуй ответить числом 🐾",
            "question": step_data["question"],
            "hint": "",
            "emotion": "thinking"
        }

    # ===== ПРАВИЛЬНО =====
    if user_number == correct:
        state["current_step"] += 1

        if state["current_step"] >= len(state["steps"]):
            return {
                "reply": "Отлично! 🎉",
                "question": "",
                "hint": "",
                "emotion": "proud"
            }

        next_step = state["steps"][state["current_step"]]
        next_question = next_step["question"]

        # защита от повтора
        if next_question == state["last_question"]:
            state["current_step"] += 1

            if state["current_step"] >= len(state["steps"]):
                return {
                    "reply": "Отлично! 🎉",
                    "question": "",
                    "hint": "",
                    "emotion": "proud"
                }

            next_step = state["steps"][state["current_step"]]
            next_question = next_step["question"]

        state["last_question"] = next_question

        return {
            "reply": "Верно 👍",
            "question": next_question,
            "hint": "",
            "emotion": "happy"
        }

    # ===== ОШИБКА =====
    return explain_error(
        step=step_data["question"],
        user_text=text,
        correct_value=correct,
        grade=data.grade
    )