import os
import json
import re
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- OCR libs ---
try:
    import cv2
    from PIL import Image
    import pytesseract

    OCR_AVAILABLE = True
except Exception as e:
    print("OCR INIT ERROR:", e)
    OCR_AVAILABLE = False

# --- OpenAI ---
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- INIT APP ---
app = FastAPI(title="Murmatika API 🐾")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MEMORY ---
state_memory = {}

# --- MODELS ---
class ChatMessageRequest(BaseModel):
    name: str
    grade: int
    message: str
    user_id: str


# =========================
# UTILS
# =========================

def extract_number(text: str):
    nums = re.findall(r'\d+', text)
    return int(nums[-1]) if nums else None


def is_simple_addition(text: str):
    return re.match(r'^\d+\s*\+\s*\d+$', text.strip())


# =========================
# STEP ENGINE (без GPT)
# =========================

def build_addition_steps(text):
    a, b = map(int, re.findall(r'\d+', text))

    to10 = 10 - a

    if b > to10:
        rest = b - to10
        return [
            {"q": f"Сколько нужно добавить к {a}, чтобы получить 10?", "a": to10},
            {"q": f"Сколько останется от {b}, если взять {to10}?", "a": rest},
            {"q": f"Сколько будет 10 + {rest}?", "a": 10 + rest},
        ]

    return [
        {"q": f"Сколько будет {a} + {b}?", "a": a + b}
    ]


# =========================
# OCR SAFE VERSION
# =========================

def safe_ocr(image_bytes):
    if not OCR_AVAILABLE:
        return "OCR не доступен (нет библиотек)"

    try:
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return "Не удалось прочитать изображение"

        pil_img = Image.fromarray(img)

        # 🔥 если tesseract не в PATH — укажи вручную:
        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        text = pytesseract.image_to_string(pil_img)

        if not text.strip():
            return "Текст не распознан"

        return text

    except Exception as e:
        return f"OCR ERROR: {str(e)}"


# =========================
# ROUTES
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}


# -------- OCR --------
@app.post("/api/ocr")
async def ocr_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        text = safe_ocr(contents)

        # разбиваем грубо
        tasks = [t.strip() for t in text.split("\n") if t.strip()]

        if not tasks:
            tasks = [text]

        return {
            "tasks": tasks
        }

    except Exception as e:
        return {
            "tasks": [],
            "error": str(e)
        }


# -------- CHAT --------
@app.post("/api/chat/message")
def chat(data: ChatMessageRequest):
    user_id = data.user_id
    text = data.message.strip()

    # === НОВАЯ ЗАДАЧА ===
    if user_id not in state_memory:

        if is_simple_addition(text):
            steps = build_addition_steps(text)
        else:
            return {
                "reply": "Пока умею решать простые примеры 😊",
                "question": "",
                "hint": "",
                "emotion": "thinking"
            }

        state_memory[user_id] = {
            "steps": steps,
            "step": 0
        }

        return {
            "reply": "Давай решим вместе 🐾",
            "question": steps[0]["q"],
            "hint": "",
            "emotion": "thinking"
        }

    state = state_memory[user_id]
    step = state["steps"][state["step"]]

    user_num = extract_number(text)

    if user_num is None:
        return {
            "reply": "Напиши число 😊",
            "question": step["q"],
            "hint": "",
            "emotion": "thinking"
        }

    # === ПРАВИЛЬНО ===
    if user_num == step["a"]:
        state["step"] += 1

        if state["step"] >= len(state["steps"]):
            state_memory.pop(user_id)

            return {
                "reply": "Отлично! 🎉",
                "question": "",
                "hint": "",
                "emotion": "proud"
            }

        next_step = state["steps"][state["step"]]

        return {
            "reply": "Верно 👍",
            "question": next_step["q"],
            "hint": "",
            "emotion": "happy"
        }

    # === НЕПРАВИЛЬНО ===
    return {
        "reply": "Попробуй ещё 🐾",
        "question": step["q"],
        "hint": "",
        "emotion": "thinking"
    }


# -------- SOLUTION --------
@app.get("/api/solution")
def solution():
    return {
        "solution": "Решение скоро будет доступно 🐾"
    }