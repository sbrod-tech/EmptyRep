import os
import base64
import json
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI

print("🔥 FINAL WORKING MAIN LOADED")

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# SESSION (простая память)
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
# VISION (СТАБИЛЬНЫЙ JSON)
# =========================

@app.post("/api/vision")
async def vision(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
Return valid JSON only.

Find all math problems in the image.

Format:
{
  "tasks": ["problem 1", "problem 2"]
}
"""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract math problems"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        )

        content = response.choices[0].message.content
        print("VISION RAW:", content)

        return json.loads(content)

    except Exception as e:
        print("VISION ERROR:", str(e))
        return {"tasks": [], "error": str(e)}


# =========================
# РЕШЕНИЕ (СКРЫТОЕ)
# =========================

@app.get("/api/solution")
def get_solution():
    return {"solution": session["solution"]}


# =========================
# RESET
# =========================

@app.delete("/api/reset")
def reset():
    session["task"] = ""
    session["solution"] = ""
    session["steps"] = []
    session["current_step"] = 0
    return {"status": "reset"}


# =========================
# ГЕНЕРАЦИЯ ШАГОВ
# =========================

def generate_steps(task):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
Ты учитель начальной школы.

Разбей задачу на шаги.

Верни JSON:

{
  "steps": [
    {
      "question": "...",
      "answer": "...",
      "hint": "..."
    }
  ]
}

ВАЖНО:
- question = вопрос ученику
- answer = точный ответ
- hint = подсказка без решения
"""
            },
            {
                "role": "user",
                "content": task
            }
        ]
    )

    content = response.choices[0].message.content
    print("STEPS RAW:", content)

    return json.loads(content)["steps"]


# =========================
# ПОЛНОЕ РЕШЕНИЕ
# =========================

def generate_solution(task):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Реши задачу подробно по шагам."},
            {"role": "user", "content": task}
        ]
    )

    return response.choices[0].message.content


# =========================
# ЧАТ
# =========================

@app.post("/api/chat/message")
def chat(req: ChatRequest):

    user = req.message.lower().strip()

    # === старт новой задачи ===
    if session["task"] == "":
        session["task"] = req.message

        try:
            session["solution"] = generate_solution(req.message)
            session["steps"] = generate_steps(req.message)
            session["current_step"] = 0

            first = session["steps"][0]

            return {
                "reply": f"Давай разберём вместе 🐾\n\n{first['question']}",
                "emotion": "thinking"
            }

        except Exception as e:
            print("INIT ERROR:", str(e))
            return {
                "reply": "Не удалось разобрать задачу 😢 Попробуй ещё раз",
                "emotion": "confused"
            }

    # === не понимаю ===
    if "не понимаю" in user:
        step = session["steps"][session["current_step"]]
        return {
            "reply": f"Давай проще 😊\n\n{step['hint']}",
            "emotion": "confused"
        }

    step = session["steps"][session["current_step"]]

    # === проверка ответа ===
    if user.replace(" ", "") == step["answer"].replace(" ", ""):

        session["current_step"] += 1

        # === завершено ===
        if session["current_step"] >= len(session["steps"]):
            return {
                "reply": "Отлично! 🎉 Ты решил задачу!",
                "emotion": "happy"
            }

        next_step = session["steps"][session["current_step"]]

        return {
            "reply": f"Верно 👍\n\n{next_step['question']}",
            "emotion": "happy"
        }

    else:
        return {
            "reply": f"Почти 🤔\n\nПодсказка:\n{step['hint']}",
            "emotion": "thinking"
        }