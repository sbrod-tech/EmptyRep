from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import json

app = FastAPI()
client = OpenAI()

# 🔥 хранилище сессий
sessions = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 📸 OCR → список задач
# =========================
@app.post("/api/vision")
async def vision(file: UploadFile = File(...)):
    image_bytes = await file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": f"""
Извлеки ВСЕ математические задачи с изображения.

Верни JSON:
{{
  "tasks": ["задача1", "задача2"]
}}

image: data:image/jpeg;base64,{base64_image}
"""
            }
        ],
    )

    data = json.loads(response.choices[0].message.content)
    return data


# =========================
# 🧠 РЕШЕНИЕ + ШАГИ
# =========================
def generate_solution(task: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "user",
                "content": f"""
Реши задачу и обучай ребёнка (4 класс).

Задача:
{task}

Верни JSON:

{{
  "intro": "кратко объясни с чего начать",
  "solution": "полное решение",
  "steps": [
    {{
      "question": "вопрос",
      "answer": "ответ",
      "hint": "подсказка"
    }}
  ]
}}

ВАЖНО:
- сначала реши задачу
- затем разбей на шаги
- шаги должны быть конкретными
"""
            }
        ],
    )

    return json.loads(response.choices[0].message.content)


# =========================
# 💬 ЧАТ
# =========================
@app.post("/api/chat/message")
async def chat(data: dict):
    message = data.get("message")
    session_id = data.get("session_id")

    if not session_id:
        return {"reply": "Ошибка: нет session_id"}

    # =========================
    # 🚀 ПЕРВЫЙ ЗАПУСК
    # =========================
    if session_id not in sessions:
        task = message

        generated = generate_solution(task)

        sessions[session_id] = {
            "task": task,
            "solution": generated["solution"],
            "steps": generated["steps"],
            "intro": generated["intro"],
            "current_step": 0
        }

        first_step = generated["steps"][0]

        return {
            "reply": f"{generated['intro']}\n\n👉 {first_step['question']}",
            "hint": first_step["hint"],
            "emotion": "thinking"
        }

    # =========================
    # 📚 ПРОДОЛЖЕНИЕ
    # =========================
    session = sessions[session_id]
    steps = session["steps"]
    i = session["current_step"]

    if i >= len(steps):
        return {
            "reply": "🎉 Задача решена! Нажми кнопку, чтобы посмотреть решение.",
            "emotion": "happy"
        }

    current = steps[i]

    # =========================
    # ✅ ПРОВЕРКА ОТВЕТА
    # =========================
    user_answer = message.strip().lower()
    correct = current["answer"].strip().lower()

    if user_answer == correct:
        session["current_step"] += 1

        if session["current_step"] >= len(steps):
            return {
                "reply": "🔥 Отлично! Ты решил задачу!",
                "emotion": "happy"
            }

        next_step = steps[session["current_step"]]

        return {
            "reply": f"Верно 👍\n\n👉 {next_step['question']}",
            "hint": next_step["hint"],
            "emotion": "happy"
        }

    else:
        return {
            "reply": f"Не совсем так 🤔\n\n👉 {current['question']}",
            "hint": current["hint"],
            "emotion": "confused"
        }


# =========================
# 📖 РЕШЕНИЕ
# =========================
@app.get("/api/solution/{session_id}")
def get_solution(session_id: str):
    session = sessions.get(session_id)

    if not session:
        return {"solution": "Сессия не найдена"}

    return {"solution": session["solution"]}


# =========================
# 🔄 RESET
# =========================
@app.delete("/api/reset/{session_id}")
def reset(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "ok"}


# =========================
# ❤️ HEALTH
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}