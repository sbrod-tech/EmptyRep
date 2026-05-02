from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import json
import re
from openai import OpenAI

app = FastAPI()

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

client = OpenAI()

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

    tasks = [t for t in tasks if len(t) > 40]
    tasks = [t for t in tasks if re.search(r'[а-яА-Я]', t)]

    return tasks

# =========================
# 🔢 SIMPLE EXPRESSION
# =========================
def is_simple_expression(task: str):
    return bool(re.match(r'^[\d\s\+\-\*/\(\)]+$', task.strip()))

def generate_simple_steps(task: str):
    task = task.replace(" ", "")

    if "+" in task:
        a, b = map(int, task.split("+"))

        a10, a1 = a // 10 * 10, a % 10
        b10, b1 = b // 10 * 10, b % 10

        sum10 = a10 + b10
        sum1 = a1 + b1
        result = sum10 + sum1

        return {
            "steps": [
                {"question": f"Разложим {a}. Сколько это?", "answer": f"{a10}+{a1}", "hint": "Раздели на десятки и единицы"},
                {"question": f"Разложим {b}. Сколько это?", "answer": f"{b10}+{b1}", "hint": "То же самое"},
                {"question": f"Сколько будет {a10} + {b10}?", "answer": str(sum10), "hint": "Складываем десятки"},
                {"question": f"Сколько будет {a1} + {b1}?", "answer": str(sum1), "hint": "Складываем единицы"},
                {"question": f"Сколько будет {sum10} + {sum1}?", "answer": str(result), "hint": "Финальный шаг"}
            ]
        }

    if "-" in task:
        a, b = map(int, task.split("-"))
        return {
            "steps": [
                {"question": f"Сколько будет {task}?", "answer": str(a - b), "hint": "Попробуй вычесть"}
            ]
        }

    return {
        "steps": [
            {"question": f"Сколько будет {task}?", "answer": "", "hint": ""}
        ]
    }

# =========================
# 🧠 СНАЧАЛА РЕШАЕМ
# =========================
def solve_task_first(task: str):
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
{{
  "final_answer": "...",
  "plan": ["шаг1","шаг2","шаг3"]
}}
"""
        }]
    )
    return json.loads(res.choices[0].message.content)

# =========================
# 🧠 ШАГИ ПО ПЛАНУ
# =========================
def generate_steps_from_plan(task: str, solution: dict):
    plan = solution.get("plan", [])

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": f"""
Ты репетитор.

ПЛАН:
{plan}

Сделай обучающие шаги.

ПРАВИЛА:
- каждый шаг = конкретное действие
- простой язык
- без фантазии
- не менять числа

Верни JSON steps.

Задача:
{task}
"""
        }]
    )

    return json.loads(res.choices[0].message.content)

# =========================
# 🔒 ПРОВЕРКИ
# =========================
def contains_new_numbers(task, steps):
    task_nums = set(re.findall(r'\d+', task))
    step_nums = set(re.findall(r'\d+', json.dumps(steps)))
    return not step_nums.issubset(task_nums)

def is_bad_step(step):
    q = step["question"].lower()
    bad = ["что такое", "как думаешь", "например"]
    return any(b in q for b in bad)

def validate_steps(task, steps):
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": f"""
Проверь шаги.

Задача:
{task}

Шаги:
{json.dumps(steps, ensure_ascii=False)}

Ответ:
{{ "valid": true }} или {{ "valid": false }}
"""
            }]
        )
        return json.loads(res.choices[0].message.content)
    except:
        return {"valid": True}

# =========================
# 📸 OCR
# =========================
@app.post("/api/vision")
async def vision(file: UploadFile = File(...)):
    try:
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
    except:
        return {"text": ""}

# =========================
# ✂️ SPLIT
# =========================
@app.post("/api/split")
async def split(req: TextRequest):
    try:
        text = clean_text(req.text)
        return {"tasks": split_tasks_smart(text)}
    except:
        return {"tasks": [req.text]}

# =========================
# 🚀 GENERATE
# =========================
@app.post("/api/generate")
async def generate(req: TaskRequest):
    try:
        task = req.task.strip()

        # примеры
        if is_simple_expression(task):
            return generate_simple_steps(task)

        # сначала решаем
        solution = solve_task_first(task)

        # потом обучаем
        data = generate_steps_from_plan(task, solution)

        # защиты
        if contains_new_numbers(task, data):
            data = generate_steps_from_plan(task, solution)

        if any(is_bad_step(s) for s in data["steps"]):
            data = generate_steps_from_plan(task, solution)

        val = validate_steps(task, data["steps"])
        if not val.get("valid", True):
            data = generate_steps_from_plan(task, solution)

        return data

    except:
        return {
            "steps": [
                {
                    "question": "Ошибка 😢",
                    "answer": "",
                    "hint": ""
                }
            ]
        }