import os
import json
import re
import cv2
import numpy as np
import pytesseract

from PIL import Image
from fastapi import FastAPI, UploadFile, File
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

def is_new_problem(text: str):
    t = text.lower()
    return (
        len(text) > 20 and
        any(w in t for w in [
            "сколько", "найди", "во сколько", "шел", "минут"
        ])
    )

def extract_number(text: str):
    nums = re.findall(r'\d+', text)
    return int(nums[-1]) if nums else None

def is_help_request(text: str):
    t = text.lower()
    return any(w in t for w in [
        "как", "почему", "объясни", "не понимаю"
    ])

# ================= VISUAL EXPLAIN =================

def detect_concept(text: str):
    if "+" in text:
        return "addition"
    if "-" in text:
        return "subtraction"
    return "general"

def visual_explain(concept, text):
    nums = re.findall(r'\d+', text)

    if concept == "addition" and len(nums) >= 2:
        a, b = int(nums[0]), int(nums[1])
        return f"""
Давай покажу на палочках 🐾

{a}: {"|"*a}
{b}: {"|"*b}

Теперь сложи все палочки 😊
"""

    return None

def explain_concept(user_text, current_step, grade):
    visual = visual_explain(detect_concept(user_text), user_text)

    if visual:
        return {
            "reply": visual.strip(),
            "question": current_step,
            "hint": "Попробуй теперь сам 😊",
            "emotion": "thinking"
        }

    return {
        "reply": "Давай подумаем вместе 🐾",
        "question": current_step,
        "hint": "",
        "emotion": "thinking"
    }

# ================= OCR =================

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh

def clean_text(text: str):
    text = text.replace("\n", " ")
    text = re.sub(r'[^0-9а-яА-Яa-zA-Z\+\-\*/=.,() ]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_tasks_smart(text: str):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
Раздели текст на задачи.

Верни JSON:
{
  "tasks": ["..."]
}
"""
                },
                {"role": "user", "content": text}
            ]
        )

        return json.loads(response.choices[0].message.content)["tasks"]

    except:
        return [text]

# ================= STEP GENERATION =================

def is_simple_addition(text: str):
    return re.match(r'^\d+\s*\+\s*\d+$', text.strip())

def build_addition_steps(text):
    a, b = map(int, re.findall(r'\d+', text))
    to10 = 10 - a

    if b > to10:
        rest = b - to10
        return [
            {"question": f"Сколько нужно добавить к {a}, чтобы было 10?", "answer": to10},
            {"question": f"Сколько осталось от {b}?", "answer": rest},
            {"question": f"Сколько будет 10 + {rest}?", "answer": 10 + rest},
        ]

    return [
        {"question": f"Сколько будет {a} + {b}?", "answer": a + b}
    ]

def generate_steps(problem, grade):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": """
Разбей задачу на шаги.

Формат:
{
  "steps": [
    {"question": "...", "answer": 5}
  ]
}
"""
                },
                {"role": "user", "content": problem}
            ]
        )

        return json.loads(response.choices[0].message.content)["steps"]

    except:
        return [{"question": "Попробуй понять задачу", "answer": 0}]

# ================= ROUTES =================

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- OCR ----------
@app.post("/api/ocr")
async def ocr_image(file: UploadFile = File(...)):
    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    processed = preprocess_image(img)

    pil_img = Image.fromarray(processed)

    raw_text = pytesseract.image_to_string(pil_img, lang="rus+eng")
    cleaned = clean_text(raw_text)
    tasks = split_tasks_smart(cleaned)

    return {
        "tasks": tasks
    }

# ---------- CHAT ----------
@app.post("/api/chat/message")
def chat(data: ChatMessageRequest):
    user_id = data.user_id
    text = data.message.strip()

    # NEW PROBLEM
    if user_id not in state_memory or is_new_problem(text):

        if is_simple_addition(text):
            steps = build_addition_steps(text)
        else:
            steps = generate_steps(text, data.grade)

        state_memory[user_id] = {
            "steps": steps,
            "current_step": 0
        }

        return {
            "reply": "Давай решим вместе 🐾",
            "question": steps[0]["question"],
            "hint": "",
            "emotion": "thinking"
        }

    state = state_memory[user_id]
    step = state["steps"][state["current_step"]]

    # HELP
    if is_help_request(text):
        return explain_concept(
            user_text=text,
            current_step=step["question"],
            grade=data.grade
        )

    user_number = extract_number(text)

    if user_number is None:
        return {
            "reply": "Ответь числом 😊",
            "question": step["question"],
            "hint": "",
            "emotion": "thinking"
        }

    if user_number == step["answer"]:
        state["current_step"] += 1

        if state["current_step"] >= len(state["steps"]):
            return {
                "reply": "Отлично! 🎉",
                "question": "",
                "hint": "",
                "emotion": "proud"
            }

        next_step = state["steps"][state["current_step"]]

        return {
            "reply": "Верно 👍",
            "question": next_step["question"],
            "hint": "",
            "emotion": "happy"
        }

    return {
        "reply": "Попробуй ещё 🐾",
        "question": step["question"],
        "hint": "",
        "emotion": "thinking"
    }

# ---------- SOLUTION ----------
@app.get("/api/solution")
def solution():
    return {"solution": "Решение пока в разработке 🐾"}