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

Разбей задачу на шаги.

Типы:
- NUMBER (число)
- OPERATION (действие)
- TEXT (объяснение)

НЕ объясняй решение.

Верни JSON:

{
  "answer": 26,
  "steps": [
    {"question": "...", "type": "NUMBER"},
    {"question": "...", "type": "OPERATION"}
  ],
  "strategy": "..."
}
"""
            },
            {"role": "user", "content": problem}
        ]
    )

    return json.loads(response.choices[0].message.content)

# ================= ANALYZE =================

def analyze_answer(state, user_text, current_step):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
Ты учитель.

Проверь:
1. Ответ относится к текущему вопросу
2. Он логически верный

НЕ давай правильный ответ.

Верни:

{
  "correct": true/false,
  "relevant": true/false,
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

Вопрос:
{current_step["question"]}

Тип:
{current_step["type"]}

Ответ ученика:
{user_text}
"""
            }
        ]
    )

    return json.loads(response.choices[0].message.content)

# ================= HELP =================

def explain_step(state, current_step):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
Объясни текущий шаг.

НЕ решай задачу.
НЕ давай ответ.

Верни:
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

Шаг:
{current_step["question"]}
"""
            }
        ]
    )

    data = json.loads(response.choices[0].message.content)

    return {
        "reply": data["explain"],
        "question": current_step["question"],
        "hint": data["hint"],
        "emotion": "thinking"
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
            "strategy": solution["strategy"],
            "step": 0
        }

        first = solution["steps"][0]

        return {
            "reply": "Давай разберём вместе 🐾",
            "question": first["question"],
            "hint": f"Подумай: {solution['strategy']}",
            "emotion": "thinking"
        }

    state = state_memory[user_id]
    current_step = state["steps"][state["step"]]

    # ===== HELP =====
    if is_help(text):
        return explain_step(state, current_step)

    # ===== ПРОВЕРКА ТИПА =====
    user_type = detect_answer_type(text)

    if user_type != current_step["type"]:
        return {
            "reply": "Сейчас нужен другой тип ответа 😊",
            "question": current_step["question"],
            "hint": f"Подумай, это должно быть: {current_step['type']}",
            "emotion": "confused"
        }

    # ===== ФИНАЛ =====
    user_num = extract_number(text)

    if user_num is not None and state["step"] == len(state["steps"]) - 1:
        if abs(user_num - state["answer"]) < 0.001:
            state_memory.pop(user_id)
            return {
                "reply": "Отлично! 🎉 Ты решил задачу!",
                "question": "",
                "hint": "",
                "emotion": "proud"
            }

    # ===== АНАЛИЗ =====
    analysis = analyze_answer(state, text, current_step)

    # если не про тот шаг
    if not analysis["relevant"]:
        return {
            "reply": "Мы сейчас думаем над другим шагом 😊",
            "question": current_step["question"],
            "hint": "Сконцентрируйся на текущем вопросе",
            "emotion": "thinking"
        }

    # если верно → следующий шаг
    if analysis["correct"]:
        state["step"] += 1

        if state["step"] >= len(state["steps"]):
            state_memory.pop(user_id)
            return {
                "reply": "Супер! 🎉",
                "question": "",
                "hint": "",
                "emotion": "proud"
            }

        next_step = state["steps"][state["step"]]

        return {
            "reply": "Отлично 👍",
            "question": next_step["question"],
            "hint": analysis["hint"],
            "emotion": "happy"
        }

    # если ошибка
    return {
        "reply": analysis["feedback"],
        "question": current_step["question"],
        "hint": analysis["hint"],
        "emotion": "thinking"
    }

# ================= SOLUTION =================

@app.get("/api/solution")
def get_solution(user_id: str = Query(...)):
    state = state_memory.get(user_id)

    if not state:
        return {"solution": "Нет активной задачи"}

    return {"solution": "\n".join([s["question"] for s in state["steps"]])}

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