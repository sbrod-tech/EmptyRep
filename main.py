import os
import base64
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI

print("🔥 FINAL MAIN WITH STEPS LOADED")

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# ХРАНЕНИЕ СЕССИИ (временно)
# =========================
session = {
    "task": "",
    "solution": "",
    "steps": [],
    "current_step": 0
}

# =========================
# MODELS
# =========================

class ChatRequest(BaseModel):
    message: str
    name: str
    grade: int

# =========================
# HEALTH
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# OCR + РАЗБОР
# =========================

@app.post("/api/vision")
async def vision(file: UploadFile = File(...)):
    image_bytes = await file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Найди все задачи на изображении. Верни список задач текстом."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    )

    text = response.choices[0].message.content

    tasks = [t.strip() for t in text.split("\n") if t.strip()]

    return {"tasks": tasks}


# =========================
# ПОЛУЧИТЬ РЕШЕНИЕ (СКРЫТОЕ)
# =========================

@app.get("/api/solution")
def get_solution():
    return {"solution": session["solution"]}


# =========================
# СБРОС
# =========================

@app.delete("/api/reset")
def reset():
    session["task"] = ""
    session["solution"] = ""
    session["steps"] = []
    session["current_step"] = 0
    return {"status": "reset"}


# =========================
# СОЗДАНИЕ ШАГОВ
# =========================

def generate_steps(task):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": """
Ты учитель начальной школы.

Разбей задачу на логические шаги.

Формат:
[
  {"question": "...", "answer": "...", "hint": "..."},
  ...
]

ВАЖНО:
- вопрос простой
- ответ точный
- hint помогает, но не решает
"""
        },{
            "role": "user",
            "content": task
        }]
    )

    import json
    return json.loads(response.choices[0].message.content)


# =========================
# ПОЛНОЕ РЕШЕНИЕ (СКРЫТО)
# =========================

def generate_solution(task):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "Реши задачу подробно, по шагам."
        },{
            "role": "user",
            "content": task
        }]
    )

    return response.choices[0].message.content


# =========================
# ЧАТ
# =========================

@app.post("/api/chat/message")
def chat(req: ChatRequest):

    user = req.message.lower()

    # === если новая задача ===
    if session["task"] == "":
        session["task"] = req.message
        session["solution"] = generate_solution(req.message)
        session["steps"] = generate_steps(req.message)
        session["current_step"] = 0

        return {
            "reply": f"Давай разберём задачу вместе 🐾\n\n{session['steps'][0]['question']}",
            "emotion": "thinking"
        }

    # === если ученик не понимает ===
    if "не понимаю" in user:
        step = session["steps"][session["current_step"]]
        return {
            "reply": f"Ничего страшного 😊\n\nПодсказка:\n{step['hint']}",
            "emotion": "confused"
        }

    step = session["steps"][session["current_step"]]

    # === проверка ответа ===
    if user.replace(" ", "") == step["answer"].replace(" ", ""):

        session["current_step"] += 1

        # === если задача решена ===
        if session["current_step"] >= len(session["steps"]):
            return {
                "reply": "Отлично! 🎉 Мы решили задачу!",
                "emotion": "happy"
            }

        next_step = session["steps"][session["current_step"]]

        return {
            "reply": f"Верно 👍\n\n{next_step['question']}",
            "emotion": "happy"
        }

    else:
        return {
            "reply": f"Почти! 🤔\n\nПопробуй ещё раз.\n\nПодсказка:\n{step['hint']}",
            "emotion": "thinking"
        }