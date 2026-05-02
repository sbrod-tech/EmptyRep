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

def is_help_request(text: str):
    t = text.lower()
    return any(w in t for w in [
        "как", "почему", "объясни", "что значит",
        "не понимаю", "как сделать"
    ])

# ================= VISUAL EXPLAIN =================

def detect_concept(text: str):
    t = text.lower()
    if "+" in t or "плюс" in t:
        return "addition"
    if "-" in t or "минус" in t:
        return "subtraction"
    return "general"

def visual_explain(concept, user_text):
    if concept == "addition":
        nums = re.findall(r'\d+', user_text)
        if len(nums) >= 2:
            a, b = int(nums[0]), int(nums[1])

            sticks_a = "|" * a
            sticks_b = "|" * b

            return f"""
Давай покажу на палочках 🐾

{a} это:
{sticks_a}

{b} это:
{sticks_b}

Теперь сложим:

{sticks_a} + {sticks_b}

Посчитай все палочки 😊
"""
    return None

def explain_concept(user_text, current_step, grade):
    concept = detect_concept(user_text)
    visual = visual_explain(concept, user_text)

    if visual:
        return {
            "reply": visual.strip(),
            "question": current_step,
            "hint": "Попробуй теперь сам 😊",
            "emotion": "thinking"
        }

    # fallback GPT
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": f"""
Ты учитель {grade} класса.

Объясни просто и коротко.

Верни к вопросу:
{current_step}

JSON:
{{
  "reply": "...",
  "question": "...",
  "hint": "...",
  "emotion": "thinking"
}}
"""
                },
                {"role": "user", "content": user_text}
            ]
        )

        return json.loads(response.choices[0].message.content)

    except Exception:
        return {
            "reply": "Давай подумаем вместе 🐾",
            "question": current_step,
            "hint": "",
            "emotion": "thinking"
        }

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

Формат:
{{
  "steps": [
    {{ "question": "...", "answer": 36 }},
    {{ "question": "...", "answer": 5 }}
  ]
}}
"""
                },
                {"role": "user", "content": problem}
            ]
        )

        return json.loads(response.choices[0].message.content)["steps"]

    except Exception:
        return [{"question": "Попробуй понять задачу", "answer": 0}]

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
Ты учитель {grade} класса.

Шаг:
{step}

Ответ ученика:
{user_text}

Правильный ответ:
{correct_value}

Объясни ошибку кратко.

JSON:
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
            "hint": "",
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
            "steps": steps,
            "current_step": 0,
            "last_question": None
        }

        first_q = steps[0]["question"]
        state_memory[user_id]["last_question"] = first_q

        return {
            "reply": "Давай решим вместе 🐾",
            "question": first_q,
            "hint": "",
            "emotion": "thinking"
        }

    state = state_memory[user_id]
    step_data = state["steps"][state["current_step"]]

    # ===== HELP REQUEST =====
    if is_help_request(text):
        return explain_concept(
            user_text=text,
            current_step=step_data["question"],
            grade=data.grade
        )

    # ===== ANSWER CHECK =====
    user_number = extract_number(text)

    if user_number is None:
        return {
            "reply": "Ответь числом 😊",
            "question": step_data["question"],
            "hint": "",
            "emotion": "thinking"
        }

    correct = step_data["answer"]

    # ===== CORRECT =====
    if user_number == correct:
        state["current_step"] += 1

        if state["current_step"] >= len(state["steps"]):
            return {
                "reply": "Отлично! 🎉",
                "question": "",
                "hint": "",
                "emotion": "proud"
            }

        next_q = state["steps"][state["current_step"]]["question"]

        if next_q == state["last_question"]:
            state["current_step"] += 1
            if state["current_step"] < len(state["steps"]):
                next_q = state["steps"][state["current_step"]]["question"]

        state["last_question"] = next_q

        return {
            "reply": "Верно 👍",
            "question": next_q,
            "hint": "",
            "emotion": "happy"
        }

    # ===== ERROR =====
    return explain_error(
        step=step_data["question"],
        user_text=text,
        correct_value=correct,
        grade=data.grade
    )