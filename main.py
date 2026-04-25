from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Math Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Math Tutor Backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


class ChatMessageRequest(BaseModel):
    name: str
    grade: int
    message: str


@app.post("/api/chat/message")
def chat_message(data: ChatMessageRequest):
    return {
        "reply": f"{data.name}, отлично! Давай решим вместе.",
        "question": "Как ты думаешь, с чего лучше начать?",
        "hint": "Попробуй сначала найти десятки, а потом единицы.",
        "emotion": "thinking"
    }


from fastapi import FastAPI, UploadFile, File, Form
import base64
import os
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.post("/api/chat/image")
async def chat_image(
    name: str = Form(...),
    grade: str = Form(...),
    image: UploadFile = File(...)
):
    # читаем файл
    contents = await image.read()

    # кодируем в base64
    base64_image = base64.b64encode(contents).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"""
Ты детский репетитор по математике для {grade} класса.

ВАЖНО:
- НЕ давай ответ сразу
- Задавай наводящие вопросы
- Помогай думать
- Говори просто, как ребенку

Ответ верни строго в JSON:
{{
  "recognized_text": "...",
  "reply": "...",
  "question": "...",
  "hint": "...",
  "emotion": "thinking | encourage | success"
}}
"""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Реши задачу с фото"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500
        )

        content = response.choices[0].message.content

        # пытаемся распарсить JSON
        import json

        try:
            parsed = json.loads(content)
        except:
            parsed = {
                "recognized_text": "",
                "reply": content,
                "question": "",
                "hint": "",
                "emotion": "thinking"
            }

        return parsed

    except Exception as e:
        return {
            "reply": "Ой, что-то пошло не так 😅",
            "question": "Попробуй ещё раз?",
            "hint": str(e),
            "emotion": "thinking"
        }
        
@app.post("/api/chat/answer")
async def chat_answer(
    name: str = Form(...),
    grade: str = Form(...),
    user_answer: str = Form(...)
):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": f"""
Ты репетитор по математике для {grade} класса.

ВАЖНО:
- НЕ давай ответ сразу
- оцени ответ ученика
- если он близко → похвали
- если ошибка → мягко направь
- задай следующий вопрос

Ответ строго JSON:
{{
  "reply": "...",
  "question": "...",
  "hint": "...",
  "emotion": "thinking | encourage | success"
}}
"""
                },
                {
                    "role": "user",
                    "content": f"Ответ ученика: {user_answer}"
                }
            ],
            max_tokens=300
        )

        import json
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {
            "reply": "Давай попробуем ещё раз 🙂",
            "question": "Подумай ещё немного",
            "hint": str(e),
            "emotion": "thinking"
        }