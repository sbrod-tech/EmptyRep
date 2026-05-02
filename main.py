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

def is_help(text: str):
    text = text.lower()
    return any(w in text for w in ["не понимаю", "как", "помоги", "объясни"])

def type_hint(t):
    return {
        "NUMBER": "Напиши число, например: 10",
        "OPERATION": "Напиши действие, например: 10/2",
        "TEXT": "Ответь словами"
    }.get(t, "")

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

Разбей задачу на простые шаги.

Каждый шаг:
- понятный
- с примером
- с expected_answer

Верни JSON:

{
  "answer": 10,
  "steps": [
    {
      "question": "Сколько будет 10/2? (пример: 8/2=4)",
      "type": "NUMBER",
      "expected_answer": 5
    }
  ],
  "solution_steps": [
    "10 / 2 = 5"
  ]
}
"""
                },
                {"role": "user", "content": problem}
            ]
        )

        raw = response.choices[0].message.content
        print("SOLVER RAW:", raw)

        return json.loads(raw)

    except Exception as e:
        print("SOLVER ERROR:", str(e))
        return {
            "answer": 0,
            "steps": [{
                "question": "Попробуй написать задачу ещё раз",
                "type": "TEXT",
                "expected_answer": ""
            }],
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
            "answer": solution.get("answer", 0),
            "steps": solution.get("steps", []),
            "solution_steps": solution.get("solution_steps", []),
            "step": 0
        }

        first = state_memory[user_id]["steps"][0]

        return {
            "reply": "Давай разберём вместе 🐾",
            "question": first["question"],
            "hint": type_hint(first["type"]),
            "emotion": "thinking"
        }

    state = state_memory[user_id]
    current = state["steps"][state["step"]]

    # ===== HELP =====
    if is_help(text):
        return {
            "reply": "Давай разберёмся 👇",
            "question": current["question"],
            "hint": type_hint(current["type"]),
            "emotion": "thinking"
        }

    # ===== NUMBER =====
    if current["type"] == "NUMBER":
        num = extract_number(text)

        if num is None:
            return {
                "reply": "Нужно написать число 😊",
                "question": current["question"],
                "hint": "Например: 10",
                "emotion": "confused"
            }

        if abs(num - current["expected_answer"]) < 0.01:
            state["step"] += 1
        else:
            return {
                "reply": "Почти! Попробуй ещё 😊",
                "question": current["question"],
                "hint": "Проверь вычисления",
                "emotion": "thinking"
            }

    # ===== OPERATION =====
    elif current["type"] == "OPERATION":
        if any(op in text for op in ["+", "-", "*", "/", "x", "×"]):
            state["step"] += 1
        else:
            return {
                "reply": "Выбери действие 😊",
                "question": current["question"],
                "hint": "Сложение, деление, умножение или вычитание?",
                "emotion": "thinking"
            }

    # ===== TEXT =====
    else:
        state["step"] += 1

    # ===== NEXT =====
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
Найди ВСЕ математические задачи на изображении.

ВАЖНО:
- всегда верни JSON
- даже если одна задача

Формат:
{
  "tasks": ["..."]
}
"""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Извлеки задачи"},
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

        raw = response.choices[0].message.content
        print("VISION RAW:", raw)

        try:
            data = json.loads(raw)
            tasks = data.get("tasks", [])
        except:
            tasks = [raw]

        if not tasks:
            tasks = ["Не удалось распознать задачу. Попробуй сфотографировать ближе"]

        return {"tasks": tasks}

    except Exception as e:
        print("VISION ERROR:", str(e))
        return {
            "tasks": ["Ошибка распознавания. Попробуй ещё раз"]
        }

# ================= HEALTH =================

@app.get("/health")
def health():
    return {"status": "ok"}