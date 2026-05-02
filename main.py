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
    nums = re.findall(r'\d+(?:\.\d+)?', text.replace(',', '.'))
    return float(nums[-1]) if nums else None

def detect_answer_type(text: str):
    text = text.lower()

    if re.search(r'\d', text):
        return "NUMBER"

    if any(op in text for op in ["+", "-", "*", "x", "×", "/", "дел", "умнож", "слож"]):
        return "OPERATION"

    return "TEXT"

def is_help(text: str):
    text = text.lower()
    return any(w in text for w in [
        "не понимаю", "как", "помоги", "объясни"
    ])

def type_hint(expected_type):
    if expected_type == "NUMBER":
        return "Напиши число, например: 10"
    if expected_type == "OPERATION":
        return "Напиши действие, например: 10/2 или 5+3"
    return "Ответь словами"

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
Ты учитель начальных классов.

Разбей задачу на простые шаги.

Каждый шаг:
- понятный ребёнку
- с примером ответа
- с правильным expected_answer

Верни:

{
  "answer": 33.75,
  "steps": [
    {
      "question": "Сколько сена на одну лошадь? (пример: 10/2 = 5)",
      "type": "NUMBER",
      "expected_answer": 33.75
    }
  ],
  "solution_steps": [
    "135 / 4 = 33.75"
  ]
}
"""
                },
                {"role": "user", "content": problem}
            ]
        )

        return json.loads(response.choices[0].message.content)

    except Exception:
        # fallback
        return {
            "answer": 0,
            "steps": [
                {
                    "question": "Попробуй переформулировать задачу",
                    "type": "TEXT",
                    "expected_answer": ""
                }
            ],
            "solution_steps": []
        }

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
            "solution_steps": solution.get("solution_steps", []),
            "step": 0
        }

        first = solution["steps"][0]

        return {
            "reply": "Давай разберём вместе 🐾",
            "question": first["question"],
            "hint": type_hint(first["type"]),
            "emotion": "thinking"
        }

    state = state_memory[user_id]
    current_step = state["steps"][state["step"]]

    # ===== HELP =====
    if is_help(text):
        return {
            "reply": "Давай разберёмся 👇",
            "question": current_step["question"],
            "hint": type_hint(current_step["type"]),
            "emotion": "thinking"
        }

    # ===== ПРОВЕРКА ТИПА =====
    user_type = detect_answer_type(text)

    if user_type != current_step["type"]:
        return {
            "reply": "Попробуй по-другому 😊",
            "question": current_step["question"],
            "hint": type_hint(current_step["type"]),
            "emotion": "confused"
        }

    # ===== ПРОВЕРКА ЧИСЛА =====
    if current_step["type"] == "NUMBER":
        user_num = extract_number(text)

        if user_num is None:
            return {
                "reply": "Попробуй написать число 😊",
                "question": current_step["question"],
                "hint": "Например: 10",
                "emotion": "confused"
            }

        correct = abs(user_num - current_step["expected_answer"]) < 0.01

        if correct:
            state["step"] += 1

            # завершение
            if state["step"] >= len(state["steps"]):
                state_memory.pop(user_id)

                return {
                    "reply": "Отлично! 🎉 Ты решил задачу!",
                    "question": "",
                    "hint": "",
                    "emotion": "proud"
                }

            next_step = state["steps"][state["step"]]

            return {
                "reply": "Верно 👍",
                "question": next_step["question"],
                "hint": type_hint(next_step["type"]),
                "emotion": "happy"
            }

        else:
            return {
                "reply": "Почти! Попробуй ещё раз 😊",
                "question": current_step["question"],
                "hint": "Проверь вычисления",
                "emotion": "thinking"
            }

    # ===== OPERATION =====
    if current_step["type"] == "OPERATION":

        if any(op in text for op in ["/", "дел"]):
            state["step"] += 1
        elif any(op in text for op in ["*", "x", "×", "умнож"]):
            state["step"] += 1
        elif any(op in text for op in ["+", "слож"]):
            state["step"] += 1
        elif any(op in text for op in ["-", "выч"]):
            state["step"] += 1
        else:
            return {
                "reply": "Подумай, какое действие нужно 😊",
                "question": current_step["question"],
                "hint": "Сложение, вычитание, умножение или деление?",
                "emotion": "thinking"
            }

        next_step = state["steps"][state["step"]]

        return {
            "reply": "Хорошо 👍",
            "question": next_step["question"],
            "hint": type_hint(next_step["type"]),
            "emotion": "happy"
        }

    # ===== TEXT =====
    state["step"] += 1

    if state["step"] >= len(state["steps"]):
        state_memory.pop(user_id)

        return {
            "reply": "Отлично! 🎉",
            "question": "",
            "hint": "",
            "emotion": "proud"
        }

    next_step = state["steps"][state["step"]]

    return {
        "reply": "Хорошо 👍",
        "question": next_step["question"],
        "hint": type_hint(next_step["type"]),
        "emotion": "happy"
    }

# ================= SOLUTION =================

@app.get("/api/solution")
def get_solution(user_id: str = Query(...)):
    state = state_memory.get(user_id)

    if not state:
        return {"solution": "Нет активной задачи"}

    return {"solution": "\n".join(state["solution_steps"])}

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

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
Найди задачи на изображении.
НЕ решай.

Верни:
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

    except Exception:
        return {"tasks": []}

# ================= HEALTH =================

@app.get("/health")
def health():
    return {"status": "ok"}