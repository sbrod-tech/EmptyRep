import os
import json
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ===== MEMORY =====
state_memory = {}

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

def decompose_expression(expr):
    nums = re.findall(r'\d+', expr)
    if not nums:
        return None

    num = int(nums[-1])
    if num <= 10:
        return " + ".join(["1"] * num)
    return None

def adapt_step(step_text, mode):
    if mode == "simpler":
        return f"{step_text}\n👉 Давай упростим 🐾"
    if mode == "faster":
        return f"{step_text} (можешь решить сразу)"
    return step_text

# ===== STEP GENERATION =====

def generate_steps(problem, grade):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"""
Разбей задачу для {grade} класса на логические шаги.

Верни JSON:
{{
  "steps": ["шаг 1", "шаг 2", "шаг 3"]
}}

НЕ решай задачу.
Ответ только JSON.
"""
            },
            {"role": "user", "content": problem}
        ]
    )

    return json.loads(response.choices[0].message.content)["steps"]

# ===== MODEL =====

class ChatMessageRequest(BaseModel):
    name: str
    grade: int
    message: str
    user_id: str

# ===== ROUTES =====

@app.get("/health")
def health():
    return {"status": "ok"}

# ===== MAIN LOGIC =====

@app.post("/api/chat/message")
def chat(data: ChatMessageRequest):
    user_id = data.user_id

    # ===== НОВАЯ ЗАДАЧА =====
    if user_id not in state_memory or len(data.message) > 10:
        steps = generate_steps(data.message, data.grade)

        state_memory[user_id] = {
            "problem": data.message,
            "steps": steps,
            "current_step": 0,
            "mistakes": 0,
            "success_streak": 0,
            "mode": "normal"
        }

        return {
            "reply": "Давай решим задачу шаг за шагом 🐾",
            "question": steps[0],
            "hint": "",
            "emotion": "thinking"
        }

    # ===== СТАРАЯ ЗАДАЧА =====
    state = state_memory[user_id]
    step_index = state["current_step"]
    current_step = state["steps"][step_index]

    user_answer = extract_number(data.message)
    correct = safe_eval(data.message)

    # ===== ПРОВЕРКА =====
    if correct is not None and user_answer is not None:
        if user_answer == correct:
            state["success_streak"] += 1
            state["mistakes"] = 0

            if state["success_streak"] >= 2:
                state["mode"] = "faster"

            state["current_step"] += 1

            if state["current_step"] >= len(state["steps"]):
                return {
                    "reply": "Отлично! Мы решили задачу 🎉",
                    "question": "",
                    "hint": "",
                    "emotion": "proud"
                }

            next_step = state["steps"][state["current_step"]]
            next_step = adapt_step(next_step, state["mode"])

            return {
                "reply": "Верно 👍",
                "question": next_step,
                "hint": "",
                "emotion": "happy"
            }

        else:
            state["mistakes"] += 1
            state["success_streak"] = 0

            if state["mistakes"] >= 1:
                state["mode"] = "simpler"

            decomposed = decompose_expression(data.message)

            hint = "Попробуй ещё раз"
            if decomposed:
                hint = f"Попробуй так: {decomposed}"

            return {
                "reply": "Давай разберёмся 🐾",
                "question": adapt_step(current_step, state["mode"]),
                "hint": hint,
                "emotion": "thinking"
            }

    # ===== FALLBACK GPT =====
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": f"""
Ты Мурчик 🐾.

Отвечай только JSON:
{{
  "reply": "...",
  "question": "...",
  "hint": "...",
  "emotion": "thinking"
}}
"""
            },
            {"role": "user", "content": data.message}
        ]
    )

    return json.loads(response.choices[0].message.content)