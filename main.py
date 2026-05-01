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

# ================= UTILS =================

def is_new_problem(text: str):
    t = text.lower()
    return (
        len(text) > 20 and
        any(w in t for w in [
            "сколько", "во сколько", "найди",
            "пошел", "встретились", "вышел"
        ])
    )

def extract_number(text: str):
    nums = re.findall(r'\d+', text)
    return int(nums[-1]) if nums else None

def parse_expression(text: str):
    match = re.search(r'(\d+)\s*([\+\-\*/])\s*(\d+)', text)
    if not match:
        return None

    a, op, b = match.groups()
    expr = f"{a}{op}{b}"

    try:
        result = eval(expr)
    except:
        return None

    return expr, result

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
Разбей задачу для ученика {grade} класса на ЧЁТКИЕ шаги.

ПРАВИЛА:
- каждый шаг = 1 вопрос
- без решения
- без лишнего текста

Пример:
[
 "Во сколько Ваня вышел?",
 "Сколько минут шел Коля?"
]

Ответ строго JSON:
{{ "steps": ["...", "..."] }}
"""
                },
                {"role": "user", "content": problem}
            ]
        )

        return json.loads(response.choices[0].message.content)["steps"]

    except Exception:
        # fallback если GPT не ответил
        return ["Попробуй понять условие задачи"]

# ================= GPT FORMATTER =================

def build_murchik(step, is_correct, grade):
    status = "правильно" if is_correct else "ошибка"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": f"""
Ты Мурчик 🐾 — учитель начальных классов.

Текущий шаг:
{step}

Ответ ученика: {status}

Сформируй:
- короткую реакцию (1 строка)
- ОДИН вопрос
- ОДНУ подсказку

НЕ уходи от шага.

Ответ строго JSON:
{{
  "reply": "...",
  "question": "...",
  "hint": "...",
  "emotion": "happy | thinking | confused | proud"
}}
"""
                }
            ]
        )

        return json.loads(response.choices[0].message.content)

    except Exception:
        # fallback если GPT не ответил
        return {
            "reply": "Давай подумаем вместе 🐾",
            "question": step,
            "hint": "",
            "emotion": "thinking"
        }

# ================= MODEL =================

class ChatMessageRequest(BaseModel):
    name: str
    grade: int
    message: str
    user_id: str

# ================= ROUTES =================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat/message")
def chat(data: ChatMessageRequest):
    user_id = data.user_id
    text = data.message.strip()

    # ================= NEW PROBLEM =================

    if user_id not in state_memory or is_new_problem(text):

        steps = generate_steps(text, data.grade)

        state_memory[user_id] = {
            "problem": text,
            "steps": steps,
            "current_step": 0,
            "expected_answer": None
        }

        return {
            "reply": "Давай решим задачу вместе 🐾",
            "question": steps[0],
            "hint": "",
            "emotion": "thinking"
        }

    # ================= CURRENT STATE =================

    state = state_memory[user_id]
    step_index = state["current_step"]
    current_step = state["steps"][step_index]

    parsed_expr = parse_expression(text)
    user_number = extract_number(text)

    is_correct = None

    # ================= USER INPUT = EXPRESSION =================

    if parsed_expr:
        expr, result = parsed_expr
        state["expected_answer"] = result
        is_correct = True

    # ================= USER INPUT = NUMBER =================

    elif user_number is not None and state["expected_answer"] is not None:
        is_correct = (user_number == state["expected_answer"])

    # ================= CORRECT =================

    if is_correct is True:
        state["current_step"] += 1

        if state["current_step"] >= len(state["steps"]):
            return {
                "reply": "Отлично! Мы решили задачу 🎉",
                "question": "",
                "hint": "",
                "emotion": "proud"
            }

        next_step = state["steps"][state["current_step"]]

        return build_murchik(next_step, True, data.grade)

    # ================= WRONG =================

    if is_correct is False:
        return build_murchik(current_step, False, data.grade)

    # ================= UNKNOWN =================

    return {
        "reply": "Давай подумаем вместе 🐾",
        "question": current_step,
        "hint": "Попробуй ответить числом",
        "emotion": "thinking"
    }