import os
import json
import base64
import re

from fastapi import FastAPI, UploadFile, File, Query
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

state_memory = {}

# ================= MODELS =================

class ChatMessageRequest(BaseModel):
    name: str
    grade: int
    message: str
    user_id: str

# ================= UTILS =================

def extract_number(text: str):
    nums = re.findall(r'\d+(?:\.\d+)?', text)
    return float(nums[-1]) if nums else None

def is_help(text: str):
    text = text.lower()
    return any(w in text for w in [
        "не понимаю", "непонятно", "как", "помоги", "объясни"
    ])

# ================= SOLVER =================

def solve_problem(problem: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
Ты учитель начальных классов.

Реши задачу ТОЛЬКО ДЛЯ СЕБЯ.
Создай план обучения.

НЕ объясняй решение напрямую.

Верни JSON:

{
  "answer": 26,
  "steps": ["9+10=19", "19+7=26"],
  "questions": [
    "Что известно в задаче?",
    "Что нужно найти?",
    "Какое действие подойдёт?"
  ],
  "strategy": "сложение"
}
"""
            },
            {"role": "user", "content": problem}
        ]
    )

    return json.loads(response.choices[0].message.content)

# ================= HELP EXPLAIN =================

def explain_step(state):
    current_q = state["questions"][state["step"]]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
Ты учитель 1-5 класса.

Объясни ТОЛЬКО текущий шаг.

СТРОГО:
- не решай задачу
- не давай финальный ответ
- не меняй задачу
- объясняй просто

Верни JSON:

{
  "explain": "...",
  "hint": "..."
}
"""
            },
            {
                "role": "user",
                "content": f"""
Задача:
{state["problem"]}

Текущий шаг:
{current_q}
"""
            }
        ]
    )

    data = json.loads(response.choices[0].message.content)

    return {
        "reply": data["explain"],
        "question": current_q,
        "hint": data["hint"],
        "emotion": "thinking"
    }

# ================= ANALYZE =================

def analyze_answer(state, user_text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
Ты учитель младших классов.

Проанализируй ответ ученика.

СТРОГО:
- не давай правильный ответ
- оцени мышление
- если почти верно → поддержи
- если ошибка → мягко направь

Верни JSON:

{
  "correct": true/false,
  "feedback": "...",
  "hint": "..."
}
"""
            },
            {
                "role": "user",
                "content": f"""
Задача:
{state["problem"]}

Правильный ответ:
{state["answer"]}

Ответ ученика:
{user_text}
"""
            }
        ]
    )

    return json.loads(response.choices[0].message.content)

# ================= CHAT =================

@app.post("/api/chat/message")
def chat(data: ChatMessageRequest):
    user_id = data.user_id
    text = data.message.strip()

    # ===== НОВАЯ ЗАДАЧА =====
    if user_id not in state_memory:

        solution = solve_problem(text)

        state_memory[user_id] = {
            "problem": text,
            "answer": solution["answer"],
            "steps": solution["steps"],
            "questions": solution["questions"],
            "strategy": solution["strategy"],
            "step": 0
        }

        return {
            "reply": "Давай разберём вместе 🐾",
            "question": solution["questions"][0],
            "hint": f"Подумай: {solution['strategy']}",
            "emotion": "thinking"
        }

    state = state_memory[user_id]

    # ===== HELP =====
    if is_help(text):
        return explain_step(state)

    # ===== ЧИСЛО =====
    user_num = extract_number(text)

    if user_num is not None:
        if abs(user_num - state["answer"]) < 0.001:
            state_memory.pop(user_id)
            return {
                "reply": "Отлично! 🎉 Ты решил задачу!",
                "question": "",
                "hint": "",
                "emotion": "proud"
            }

    # ===== АНАЛИЗ =====
    analysis = analyze_answer(state, text)

    # продвигаем шаг ТОЛЬКО тут
    state["step"] = min(state["step"] + 1, len(state["questions"]) - 1)

    return {
        "reply": analysis["feedback"],
        "question": state["questions"][state["step"]],
        "hint": analysis["hint"],
        "emotion": "thinking"
    }

# ================= SOLUTION =================

@app.get("/api/solution")
def get_solution(user_id: str = Query(...)):
    state = state_memory.get(user_id)

    if not state:
        return {"solution": "Нет активной задачи"}

    return {"solution": "\n".join(state["steps"])}

# ================= RESET =================

@app.delete("/api/reset")
def reset(user_id: str):
    state_memory.pop(user_id, None)
    return {"status": "reset"}

# ================= VISION =================

@app.post("/api/vision")
async def vision(file: UploadFile = File(...)):
    contents = await file.read()
    base64_image = base64.b64encode(contents).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
Найди задачи на изображении.
НЕ решай.

Верни JSON:
{
  "tasks": ["..."]
}
"""
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Найди задачи"},
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
    return {"tasks": data.get("tasks", [])}

# ================= HEALTH =================

@app.get("/health")
def health():
    return {"status": "ok"}