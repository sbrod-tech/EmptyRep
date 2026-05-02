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

# ================= MEMORY =================

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

# ================= SOLVER =================

def solve_problem(problem: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
Ты учитель младших классов.

Реши задачу, НО НЕ объясняй её ученику.

Также создай обучающие вопросы.

Верни JSON:

{
  "answer": 26,
  "steps": ["..."],
  "questions": [
    "Что известно в задаче?",
    "Что нужно найти?",
    "Какое действие подойдёт?"
  ],
  "strategy": "сложение / деление / логика"
}
"""
                },
                {"role": "user", "content": problem}
            ]
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "answer": None,
            "steps": [],
            "questions": ["Давай разберём задачу 😊"],
            "strategy": "unknown",
            "error": str(e)
        }

# ================= ANALYZE =================

def analyze_step(problem, user_input, correct_answer, steps):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
Ты учитель младших классов.

Проанализируй ответ ученика.

ВАЖНО:
- не говори сразу правильный ответ
- оцени ход мысли
- дай мягкую подсказку

Верни JSON:

{
  "is_correct": true/false,
  "feedback": "объяснение",
  "next_hint": "подсказка"
}
"""
                },
                {
                    "role": "user",
                    "content": f"""
Задача: {problem}

Правильный ответ: {correct_answer}

Решение:
{steps}

Ответ ученика:
{user_input}
"""
                }
            ]
        )

        return json.loads(response.choices[0].message.content)

    except Exception:
        return {
            "is_correct": False,
            "feedback": "Давай подумаем ещё 😊",
            "next_hint": ""
        }

# ================= VISION =================

@app.post("/api/vision")
async def vision(file: UploadFile = File(...)):
    try:
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

    except Exception as e:
        return {"tasks": [], "error": str(e)}

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
            "answer": solution.get("answer"),
            "steps": solution.get("steps", []),
            "questions": solution.get("questions", []),
            "strategy": solution.get("strategy", ""),
            "step": 0
        }

        return {
            "reply": "Давай разберём вместе 🐾",
            "question": state_memory[user_id]["questions"][0],
            "hint": f"Подумай: {solution.get('strategy')}",
            "emotion": "thinking"
        }

    # ===== ПРОДОЛЖЕНИЕ =====
    state = state_memory[user_id]

    analysis = analyze_step(
        state["problem"],
        text,
        state["answer"],
        state["steps"]
    )

    # ===== ЕСЛИ ПРАВИЛЬНО =====
    if analysis["is_correct"]:
        state_memory.pop(user_id)

        return {
            "reply": "Отлично! 🎉 " + analysis["feedback"],
            "question": "",
            "hint": "",
            "emotion": "proud"
        }

    # ===== ЕСЛИ ОШИБКА =====
    questions = state["questions"]

    state["step"] = min(state["step"] + 1, len(questions) - 1)

    return {
        "reply": analysis["feedback"],
        "question": questions[state["step"]],
        "hint": analysis["next_hint"],
        "emotion": "thinking"
    }

# ================= SOLUTION =================

@app.get("/api/solution")
def get_solution(user_id: str = Query(...)):
    state = state_memory.get(user_id)

    if not state:
        return {"solution": "Нет активной задачи"}

    return {
        "solution": "\n".join(state.get("steps", []))
    }

# ================= RESET =================

@app.delete("/api/reset")
def reset(user_id: str):
    state_memory.pop(user_id, None)
    return {"status": "reset"}

# ================= HEALTH =================

@app.get("/health")
def health():
    return {"status": "ok"}