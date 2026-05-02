import os
import json
import base64
import re

from fastapi import FastAPI, UploadFile, File, Query
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

state_memory = {}

# ================= MODELS =================

class ChatMessageRequest(BaseModel):
    name: str
    grade: int
    message: str
    user_id: str

# ================= UTILS =================

def extract_number(text):
    nums = re.findall(r'\d+(?:\.\d+)?', text.replace(',', '.'))
    return float(nums[-1]) if nums else None

def is_help(text):
    return any(x in text.lower() for x in ["не понимаю", "как", "помоги"])

# ================= SOLVER =================

def solve_problem(problem):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
Реши задачу и разбей её на шаги.

Верни:

{
 "answer": 33.75,
 "steps": [
   {
     "question": "Сколько всего?",
     "expected": 135
   },
   {
     "question": "Сколько на одного? (пример: 10/2)",
     "expected": 33.75
   }
 ],
 "solution": "135 / 4 = 33.75"
}
"""
            },
            {"role": "user", "content": problem}
        ]
    )

    return json.loads(response.choices[0].message.content)

# ================= CHAT =================

@app.post("/api/chat/message")
def chat(data: ChatMessageRequest):
    uid = data.user_id
    text = data.message.strip()

    # === новая задача ===
    if uid not in state_memory:
        sol = solve_problem(text)

        state_memory[uid] = {
            "problem": text,
            "steps": sol["steps"],
            "answer": sol["answer"],
            "solution": sol["solution"],
            "step": 0
        }

        return {
            "reply": "Давай разберём 👇",
            "question": sol["steps"][0]["question"],
            "hint": "Подумай внимательно",
            "emotion": "thinking"
        }

    state = state_memory[uid]
    step = state["step"]
    current = state["steps"][step]

    # === помощь ===
    if is_help(text):
        return {
            "reply": "Смотри 👇",
            "question": current["question"],
            "hint": "Попробуй разбить задачу",
            "emotion": "thinking"
        }

    user_num = extract_number(text)

    if user_num is None:
        return {
            "reply": "Нужно число 😊",
            "question": current["question"],
            "hint": "Например: 10",
            "emotion": "confused"
        }

    # === проверка ===
    if abs(user_num - current["expected"]) < 0.01:
        state["step"] += 1

        # === завершение ===
        if state["step"] >= len(state["steps"]):
            state_memory.pop(uid)

            return {
                "reply": "Отлично! 🎉",
                "question": "",
                "hint": "",
                "emotion": "proud"
            }

        next_step = state["steps"][state["step"]]

        return {
            "reply": "Верно 👍",
            "question": next_step["question"],
            "hint": "",
            "emotion": "happy"
        }

    else:
        return {
            "reply": "Почти! Попробуй ещё 😊",
            "question": current["question"],
            "hint": "Проверь вычисления",
            "emotion": "thinking"
        }

# ================= SOLUTION =================

@app.get("/api/solution")
def solution(user_id: str):
    state = state_memory.get(user_id)

    if not state:
        return {"solution": "Нет решения"}

    return {"solution": state["solution"]}

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
Извлеки задачи с изображения.

{
 "tasks": ["..."]
}
"""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "задачи"},
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

        try:
            data = json.loads(raw)
            tasks = data.get("tasks", [])
        except:
            tasks = [raw]

        if not tasks:
            tasks = ["Не удалось распознать"]

        return {"tasks": tasks}

    except Exception as e:
        print("VISION ERROR:", e)
        return {"tasks": ["Ошибка"]}

# ================= RESET =================

@app.delete("/api/reset")
def reset(user_id: str):
    state_memory.pop(user_id, None)
    return {"status": "ok"}