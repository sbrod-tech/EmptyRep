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
    nums = re.findall(r'\d+', text)
    return int(nums[-1]) if nums else None

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

Реши задачу полностью.

Верни JSON:

{
  "answer": 26,
  "steps": [
    "разложим 17 на 10 и 7",
    "9 + 10 = 19",
    "19 + 7 = 26"
  ],
  "strategy": "через десяток"
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
            "strategy": "unknown",
            "error": str(e)
        }

# ================= VISION =================

@app.post("/api/vision")
async def vision_ocr(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if len(contents) > 5_000_000:
            return {"tasks": ["Фото слишком большое 📷"]}

        base64_image = base64.b64encode(contents).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
Ты анализируешь фото страницы учебника (1–5 класс).

Нужно:
- Найти все задачи
- Разделить их
- Исправить ошибки

НЕ решай задачи

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

        solution_data = solve_problem(text)

        answer = solution_data.get("answer")
        steps = solution_data.get("steps", [])
        strategy = solution_data.get("strategy", "")

        if not steps:
            steps = [f"Сколько будет {text}?"]

        state_memory[user_id] = {
            "answer": answer,
            "steps": steps,
            "step": 0,
            "strategy": strategy,
            "solution": "\n".join(steps)
        }

        return {
            "reply": "Давай разберём вместе 🐾",
            "question": steps[0],
            "hint": f"Попробуй через: {strategy}",
            "emotion": "thinking"
        }

    # ===== ПРОДОЛЖЕНИЕ =====
    state = state_memory[user_id]

    correct_answer = state["answer"]
    user_num = extract_number(text)

    if user_num is None:
        return {
            "reply": "Напиши число 😊",
            "question": state["steps"][state["step"]],
            "hint": "",
            "emotion": "thinking"
        }

    # ===== ПРАВИЛЬНЫЙ ОТВЕТ =====
    if user_num == correct_answer:
        state_memory.pop(user_id)

        return {
            "reply": "Супер! Ты решил задачу 🎉",
            "question": "",
            "hint": "",
            "emotion": "proud"
        }

    # ===== ПОДСКАЗКА ПО ШАГАМ =====
    current_step = state["steps"][state["step"]]

    state["step"] = min(state["step"] + 1, len(state["steps"]) - 1)

    return {
        "reply": "Попробуй так 👇",
        "question": current_step,
        "hint": f"Стратегия: {state['strategy']}",
        "emotion": "thinking"
    }

# ================= SOLUTION =================

@app.get("/api/solution")
def solution(user_id: str = Query(...)):
    state = state_memory.get(user_id)

    if not state:
        return {"solution": "Сначала выбери задачу 😊"}

    return {"solution": state.get("solution", "Нет решения")}

# ================= RESET =================

@app.delete("/api/reset")
def reset(user_id: str):
    if user_id in state_memory:
        state_memory.pop(user_id)

    return {"status": "reset"}

# ================= HEALTH =================

@app.get("/health")
def health():
    return {"status": "ok"}