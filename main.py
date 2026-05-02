from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64
import json
import re

app = FastAPI()
client = OpenAI()

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
# 📸 OCR (ТЕПЕРЬ БЕЗ РЕЗКИ)
# =========================
@app.post("/api/vision")
async def vision(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        base64_image = base64.b64encode(image_bytes).decode()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
Перепиши ВЕСЬ текст с изображения БЕЗ изменений.

Правила:
- не сокращай
- не интерпретируй
- не дели
- сохрани номера задач

Верни строго JSON:
{
  "text": "..."
}
"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }]
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        full_text = data.get("text", "")

        return {"text": full_text or "Не удалось распознать текст"}

    except Exception as e:
        print("VISION ERROR:", e)
        return {"text": "Ошибка распознавания"}


# =========================
# ✂️ SPLIT ЗАДАЧ
# =========================
@app.post("/api/split")
async def split_tasks(data: dict):
    try:
        text = data.get("text", "")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": f"""
Раздели текст на задачи.

Правила:
- каждая задача должна быть ПОЛНОЙ
- не обрезай условия
- сохраняй номера (например 208, 209)

Текст:
{text}

Верни JSON:
{{
  "tasks": ["..."]
}}
"""
            }]
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        return {"tasks": data.get("tasks", [])}

    except Exception as e:
        print("SPLIT ERROR:", e)
        return {"tasks": ["Ошибка разделения"]}


# =========================
# 🔢 НОРМАЛИЗАЦИЯ ЧИСЕЛ
# =========================
def normalize_numbers(text):
    text = text.lower().replace(",", ".")
    matches = re.findall(r"-?\d+\.?\d*", text)
    return [float(x) for x in matches]


# =========================
# 🧠 ГЕНЕРАЦИЯ РЕШЕНИЯ (БЕЗ СЛИВА ОТВЕТОВ)
# =========================
def generate_solution(task):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": f"""
Ты — добрый учитель для 4 класса.

Задача:
{task}

ВАЖНО:
- НЕ давай готовый ответ ученику
- объясняй через вопросы
- ответ храни отдельно

Верни JSON:

{{
 "intro": "...",
 "solution": "...",
 "steps": [
   {{
     "question": "...",
     "answer": "...",
     "hint": "..."
   }}
 ]
}}
"""
            }]
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        data.setdefault("intro", "Давай разберёмся")
        data.setdefault("solution", "")
        data.setdefault("steps", [])

        if not data["steps"]:
            data["steps"] = [{
                "question": "С чего начнём?",
                "answer": "",
                "hint": "Подумай внимательно"
            }]

        return data

    except Exception as e:
        print("GEN ERROR:", e)
        return {
            "intro": "Давай попробуем",
            "solution": "",
            "steps": [{
                "question": "Какое первое действие?",
                "answer": "",
                "hint": "Посмотри внимательно"
            }]
        }


# =========================
# 🤖 AI ПРОВЕРКА
# =========================
def ai_check(task, step, user, correct):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.2,
            messages=[{
                "role": "user",
                "content": f"""
Ты проверяешь ученика.

Задача:
{task}

Вопрос:
{step['question']}

Правильный ответ:
{correct}

Ответ ученика:
{user}

Верни JSON:

{{
 "status": "correct | almost | wrong",
 "explanation": "...",
 "hint": "..."
}}
"""
            }]
        )

        raw = response.choices[0].message.content
        return json.loads(raw)

    except Exception as e:
        print("AI CHECK ERROR:", e)
        return {"status": "wrong", "explanation": "Ошибка", "hint": "Попробуй ещё"}


# =========================
# 💬 ЧАТ (ОБНОВЛЁННЫЙ)
# =========================
@app.post("/api/chat/message")
async def chat(data: dict):
    try:
        message = data.get("message", "").strip()
        session_id = data.get("session_id")

        if not session_id:
            return {"reply": "Нет session_id"}

        # 🚀 старт
        if session_id not in sessions:
            gen = generate_solution(message)

            sessions[session_id] = {
                "task": message,
                "solution": gen["solution"],
                "steps": gen["steps"],
                "current_step": 0,
                "attempts": 0
            }

            step = gen["steps"][0]

            return {
                "reply": f"{gen['intro']}\n\n👉 {step['question']}",
                "hint": step["hint"],
                "emotion": "thinking"
            }

        # 📚 продолжение
        s = sessions[session_id]
        step = s["steps"][s["current_step"]]

        user_nums = normalize_numbers(message)
        correct_nums = normalize_numbers(step["answer"])

        correct = False

        if user_nums and correct_nums:
            correct = user_nums == correct_nums
        else:
            result = ai_check(s["task"], step, message, step["answer"])
            if result["status"] == "correct":
                correct = True
            elif result["status"] == "almost":
                return {
                    "reply": f"Почти 👍\n{result['explanation']}",
                    "hint": result["hint"],
                    "emotion": "thinking"
                }
            else:
                s["attempts"] += 1
                return {
                    "reply": f"Не совсем так 🤔\n{result['explanation']}",
                    "hint": result["hint"],
                    "emotion": "confused"
                }

        if correct:
            s["current_step"] += 1
            s["attempts"] = 0
        else:
            s["attempts"] += 1
            return {
                "reply": "Попробуй ещё 🙂",
                "hint": step["hint"],
                "emotion": "confused"
            }

        # 🎉 конец
        if s["current_step"] >= len(s["steps"]):
            return {
                "reply": "🔥 Ты решил задачу!",
                "emotion": "happy"
            }

        next_step = s["steps"][s["current_step"]]

        return {
            "reply": f"Верно 👍\n\n👉 {next_step['question']}",
            "hint": next_step["hint"],
            "emotion": "happy"
        }

    except Exception as e:
        print("CHAT ERROR:", e)
        return {"reply": "Ошибка сервера"}


# =========================
# 📖 РЕШЕНИЕ (СКРЫТОЕ)
# =========================
@app.get("/api/solution/{session_id}")
def solution(session_id: str):
    return {"solution": sessions.get(session_id, {}).get("solution", "Нет")}


# =========================
# 🔄 RESET
# =========================
@app.delete("/api/reset/{session_id}")
def reset(session_id: str):
    sessions.pop(session_id, None)
    return {"status": "ok"}