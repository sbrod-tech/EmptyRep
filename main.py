from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import json
import re
from openai import OpenAI

app = FastAPI()
client = OpenAI()

# =========================
# 🌍 CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 📦 MODELS
# =========================
class TextRequest(BaseModel):
    text: str

class TaskRequest(BaseModel):
    task: str

# =========================
# ❤️ HEALTH
# =========================
@app.get("/")
def root():
    return {"status": "ok"}

# =========================
# 🧹 CLEAN TEXT
# =========================
def clean_text(text: str):
    text = text.replace('\r', '\n')
    text = re.sub(r'\n+', '\n', text)
    text = text.replace('•', ' ')
    text = text.replace('·', ' ')
    text = text.replace('|', ' ')
    text = text.replace('—', '-')
    return text.strip()

# =========================
# ✂️ SMART SPLIT
# =========================
def split_tasks_smart(text: str):
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    tasks = []
    current = ""

    starters = [
        "найди","определи","запиши","реши",
        "вычисли","сколько","с какой",
        "вырежи","начерти"
    ]

    def is_new(line):
        if re.match(r'^\d{2,3}[\.\)]?\s', line):
            return True
        return any(line.lower().startswith(w) for w in starters)

    for line in lines:
        if is_new(line):
            if current:
                tasks.append(current.strip())
            current = line
        else:
            if current:
                current += " " + line

    if current:
        tasks.append(current.strip())

    tasks = [t for t in tasks if len(t) > 30]
    tasks = [t for t in tasks if re.search(r'[а-яА-Я]', t)]

    return tasks

# =========================
# 🔢 SIMPLE MATH
# =========================
def is_simple_expression(task: str):
    return bool(re.match(r'^[\d\s\+\-\*/\(\)]+$', task.strip()))

def generate_simple_steps(task: str):
    task = task.replace(" ", "")

    if "+" in task:
        try:
            a, b = map(int, task.split("+"))
            a10, a1 = a // 10 * 10, a % 10
            b10, b1 = b // 10 * 10, b % 10

            sum10 = a10 + b10
            sum1 = a1 + b1
            result = sum10 + sum1

            return {
                "steps": [
                    {"question": f"Разложим {a}. Сколько это?", "answer": f"{a10}+{a1}", "hint": "Десятки и единицы"},
                    {"question": f"Разложим {b}. Сколько это?", "answer": f"{b10}+{b1}", "hint": "То же самое"},
                    {"question": f"{a10} + {b10} = ?", "answer": str(sum10), "hint": "Складываем десятки"},
                    {"question": f"{a1} + {b1} = ?", "answer": str(sum1), "hint": "Складываем единицы"},
                    {"question": f"{sum10} + {sum1} = ?", "answer": str(result), "hint": "Финал"}
                ]
            }
        except:
            pass

    try:
        result = eval(task)
        return {
            "steps": [
                {"question": f"Сколько будет {task}?", "answer": str(result), "hint": ""}
            ]
        }
    except:
        return {"steps": []}

# =========================
# 🧠 SOLVE FIRST
# =========================
def solve_task(task: str):
    res = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": f"""
Реши задачу.

Задача:
{task}

Верни JSON:
{{"final_answer":"...","plan":["шаг1","шаг2"]}}
"""
        }]
    )
    return json.loads(res.choices[0].message.content)

# =========================
# 🧠 GENERATE STEPS
# =========================
def generate_steps(task: str, solution: dict):
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": f"""
Ты репетитор.

ПЛАН:
{solution.get("plan", [])}

Сделай пошаговое обучение.

Правила:
- не менять числа
- без фантазии
- короткие вопросы

Верни JSON steps.

Задача:
{task}
"""
        }]
    )
    return json.loads(res.choices[0].message.content)

# =========================
# 🔒 VALIDATION
# =========================
def contains_new_numbers(task, steps):
    task_nums = set(re.findall(r'\d+', task))
    step_nums = set(re.findall(r'\d+', json.dumps(steps)))
    return not step_nums.issubset(task_nums)

def is_bad_step(step):
    bad = ["что такое", "как думаешь", "например"]
    return any(b in step["question"].lower() for b in bad)

# =========================
# 📸 OCR
# =========================
@app.post("/api/vision")
async def vision(file: UploadFile = File(...)):
    img = await file.read()
    b64 = base64.b64encode(img).decode()

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Распознай текст полностью"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }]
    )

    return {"text": res.choices[0].message.content}

# =========================
# ✂️ SPLIT
# =========================
@app.post("/api/split")
async def split(req: TextRequest):
    text = clean_text(req.text)
    return {"tasks": split_tasks_smart(text)}

# =========================
# 🚀 GENERATE
# =========================
@app.post("/api/generate")
async def generate(req: TaskRequest):
    task = req.task.strip()

    try:
        # 1. simple math
        if is_simple_expression(task):
            simple = generate_simple_steps(task)
            if simple["steps"]:
                return simple

        # 2. solve
        solution = solve_task(task)

        # 3. generate steps
        data = generate_steps(task, solution)

        # 4. validate
        for _ in range(2):
            if contains_new_numbers(task, data):
                data = generate_steps(task, solution)
                continue

            if any(is_bad_step(s) for s in data["steps"]):
                data = generate_steps(task, solution)
                continue

            break

        return data

    except Exception as e:
        return {
            "steps": [
                {
                    "question": "Не смог разобрать задачу 😢",
                    "answer": "",
                    "hint": ""
                }
            ]
        }
    # =========================
# ✅ CHECK ANSWER
# =========================
class CheckAnswerRequest(BaseModel):
    user_answer: str
    correct_answer: str


def normalize_answer(text: str):
    text = text.lower().strip()
    text = re.sub(r'[^0-9\.\-]', '', text)
    return text


@app.post("/api/check_answer")
async def check_answer(req: CheckAnswerRequest):
    try:
        user = normalize_answer(req.user_answer)
        correct = normalize_answer(req.correct_answer)

        if user == correct:
            return {"correct": True}

        # fallback через GPT
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": f"""
Проверь равны ли ответы.

Ответ ученика: {req.user_answer}
Правильный ответ: {req.correct_answer}

Игнорируй единицы измерения.

Верни:
{{"correct": true/false}}
"""
            }]
        )

        data = json.loads(res.choices[0].message.content)

        return {"correct": data.get("correct", False)}

    except Exception:
        return {"correct": False}