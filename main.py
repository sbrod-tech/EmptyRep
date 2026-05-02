from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import json

app = FastAPI()
client = OpenAI()

# 🔥 Хранилище сессий
sessions = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ❤️ HEALTH
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}


# =========================
# 📸 OCR (VISION)
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
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Извлеки ВСЕ математические задачи. Верни JSON: {\"tasks\": []}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ],
                }
            ],
        )

        raw = response.choices[0].message.content
        print("VISION RAW:", raw)

        data = json.loads(raw)
        tasks = data.get("tasks", [])

        # 🔥 нормализация
        clean_tasks = []
        for t in tasks:
            if isinstance(t, dict):
                clean_tasks.append(t.get("task", ""))
            else:
                clean_tasks.append(str(t))

        if not clean_tasks:
            return {"tasks": ["Не удалось распознать задачу"]}

        return {"tasks": clean_tasks}

    except Exception as e:
        print("VISION ERROR:", e)
        return {"tasks": ["Ошибка распознавания"]}


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
  "intro": "объяснение с чего начать (1-2 предложения)",
  "solution": "полное решение",
  "steps": [
    {{
      "question": "вопрос ученику",
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

    raw = response.choices[0].message.content
    print("STEPS RAW:", raw)

    return json.loads(raw)


# =========================
# 💬 ЧАТ
# =========================
@app.post("/api/chat/message")
async def chat(data: dict):
    try:
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

        user_answer = message.strip().lower()
        correct = current["answer"].strip().lower()

        # =========================
        # ✅ ПРОВЕРКА
        # =========================
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

    except Exception as e:
        print("CHAT ERROR:", e)
        return {"reply": "Ошибка сервера 😢"}


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