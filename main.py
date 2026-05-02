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
# 📸 OCR
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
                    {"type": "text", "text": "Верни json {\"tasks\": []} извлеки задачи"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }]
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        tasks = data.get("tasks", [])

        clean = []
        for t in tasks:
            if isinstance(t, dict):
                clean.append(t.get("task", ""))
            else:
                clean.append(str(t))

        clean = [t for t in clean if t.strip()]

        return {"tasks": clean or ["Не удалось распознать задачу"]}

    except Exception as e:
        print("VISION ERROR:", e)
        return {"tasks": ["Ошибка распознавания"]}


# =========================
# 🔢 НОРМАЛИЗАЦИЯ
# =========================
def normalize_number(text):
    text = text.lower().replace(",", ".")
    match = re.findall(r"-?\d+\.?\d*", text)
    return float(match[0]) if match else None


# =========================
# 🧠 ГЕНЕРАЦИЯ
# =========================
def generate_solution(task):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": f"""
Реши задачу и обучай ребенка (4 класс)

Задача:
{task}

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

Правила:
- сначала реши
- потом шаги
- шаги простые
"""
            }]
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        data.setdefault("intro", "Давай разберёмся")
        data.setdefault("solution", "Нет решения")
        data.setdefault("steps", [])

        # защита шагов
        if not isinstance(data["steps"], list) or not data["steps"]:
            data["steps"] = [{
                "question": "С чего начнём?",
                "answer": "",
                "hint": "Подумай"
            }]

        # защита answer
        for step in data["steps"]:
            if not step.get("answer"):
                step["answer"] = "не задано"

        return data

    except Exception as e:
        print("GEN ERROR:", e)
        return {
            "intro": "Давай попробуем",
            "solution": "Ошибка",
            "steps": [{
                "question": "Какое первое действие?",
                "answer": "",
                "hint": "Посмотри внимательно"
            }]
        }


# =========================
# 🧠 AI ПРОВЕРКА
# =========================
def ai_check(task, step, user, correct):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
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

Ответ:
{user}

Верни:

{{
 "status": "correct | almost | wrong",
 "explanation": "...",
 "hint": "..."
}}
"""
            }]
        )

        raw = response.choices[0].message.content

        try:
            return json.loads(raw)
        except:
            return {"status": "wrong", "explanation": raw, "hint": "Попробуй ещё"}

    except Exception as e:
        print("AI CHECK ERROR:", e)
        return {"status": "wrong", "explanation": "Ошибка проверки", "hint": "Попробуй ещё"}


# =========================
# 💬 ЧАТ
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
            if len(message) < 5:
                return {"reply": "Пустая задача"}

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

        # 🔢 сначала число
        user_num = normalize_number(message)
        correct_num = normalize_number(step["answer"])

        if user_num is not None and correct_num is not None:
            if user_num == correct_num:
                s["current_step"] += 1
                s["attempts"] = 0
            else:
                s["attempts"] += 1
        else:
            # 🤖 fallback AI
            result = ai_check(s["task"], step, message, step["answer"])
            status = result["status"]

            if status == "correct":
                s["current_step"] += 1
                s["attempts"] = 0
            elif status == "almost":
                return {
                    "reply": f"Почти 👍\n{result['explanation']}",
                    "hint": result["hint"],
                    "emotion": "thinking"
                }
            else:
                s["attempts"] += 1
                if s["attempts"] >= 3:
                    return {
                        "reply": "Давай подскажу 🙂",
                        "hint": step["answer"],
                        "emotion": "thinking"
                    }
                return {
                    "reply": f"Не совсем так 🤔\n{result['explanation']}",
                    "hint": result["hint"],
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
# 📖 РЕШЕНИЕ
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