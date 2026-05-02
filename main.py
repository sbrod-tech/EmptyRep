from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import ast
import json
import operator
import re
from typing import Any
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


class CheckAnswerRequest(BaseModel):
    user_answer: str
    correct_answer: str


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
    text = text.replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = text.replace("•", " ")
    text = text.replace("·", " ")
    text = text.replace("|", " ")
    text = text.replace("—", "-")
    return text.strip()


# =========================
# ✂️ SMART SPLIT
# =========================
def split_tasks_smart(text: str):
    """
    Делит OCR-текст на отдельные задачи.
    Важно: если задача начинается не со слова "Реши/Сколько", а, например,
    "Колхоз отправил...", мы всё равно сохраняем первую строку как задачу.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    tasks: list[str] = []
    current = ""

    starters = [
        "найди",
        "определи",
        "запиши",
        "реши",
        "вычисли",
        "сколько",
        "с какой",
        "вырежи",
        "начерти",
        "сравни",
    ]

    def has_numbering(line: str):
        return bool(re.match(r"^\d{1,3}[\.\)]?\s", line.lower()))

    def looks_like_new_after_finished_task(line: str):
        low = line.lower()
        return any(low.startswith(word) for word in starters)

    for line in lines:
        if not current:
            current = line
            continue

        # Надёжно делим по номерам: "1.", "2)", "15 ".
        # По словам "Сколько/Реши" делим только если предыдущая задача уже закончилась вопросом,
        # иначе текстовая задача распадётся на куски.
        if has_numbering(line) or (current.rstrip().endswith("?") and looks_like_new_after_finished_task(line)):
            tasks.append(current.strip())
            current = line
        else:
            current += " " + line

    if current:
        tasks.append(current.strip())

    tasks = [task for task in tasks if len(task) > 20]
    tasks = [task for task in tasks if re.search(r"[а-яА-Я]", task)]

    return tasks


# =========================
# 🧮 SAFE ARITHMETIC
# =========================
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def is_simple_expression(task: str):
    return bool(re.match(r"^[\d\s\+\-\*/\(\)\.,]+$", task.strip()))


def _safe_eval_math(expression: str):
    expression = expression.replace(",", ".").replace(" ", "")
    if not re.match(r"^[\d\+\-\*/\(\)\.]+$", expression):
        raise ValueError("Unsupported expression")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Num):  # compatibility
            return node.n
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](_eval(node.operand))
        raise ValueError("Unsupported expression")

    tree = ast.parse(expression, mode="eval")
    result = _eval(tree)
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def _format_number(value: Any):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def generate_simple_steps(task: str):
    """
    Для коротких арифметических примеров делаем шаги без запроса к модели.
    Для текстовых задач работает generate_word_problem_steps().
    """
    expression = task.replace(" ", "")

    # Красивое разложение только для A+B, где A и B — целые числа.
    if re.match(r"^\d+\+\d+$", expression):
        a, b = map(int, expression.split("+"))
        a10, a1 = a // 10 * 10, a % 10
        b10, b1 = b // 10 * 10, b % 10
        sum10 = a10 + b10
        sum1 = a1 + b1
        result = sum10 + sum1

        return {
            "task": task,
            "final_answer": str(result),
            "full_solution": [
                f"Разложим {a}: {a10}+{a1}.",
                f"Разложим {b}: {b10}+{b1}.",
                f"Сложим десятки: {a10}+{b10}={sum10}.",
                f"Сложим единицы: {a1}+{b1}={sum1}.",
                f"Сложим результаты: {sum10}+{sum1}={result}.",
            ],
            "steps": [
                {
                    "action": f"{a10}+{a1}",
                    "question": f"Разложим {a} на десятки и единицы. Напиши разложение.",
                    "answer": f"{a10}+{a1}",
                    "hint": f"В числе {a}: {a10} — десятки, {a1} — единицы.",
                    "success": "Отлично, число разложили!",
                },
                {
                    "action": f"{b10}+{b1}",
                    "question": f"Теперь разложим {b}. Как запишем?",
                    "answer": f"{b10}+{b1}",
                    "hint": f"В числе {b}: {b10} — десятки, {b1} — единицы.",
                    "success": "Верно, второе число тоже разложили!",
                },
                {
                    "action": f"{a10}+{b10}",
                    "question": f"Сложим десятки: {a10} + {b10}. Сколько получится?",
                    "answer": str(sum10),
                    "hint": "Складываем только десятки.",
                    "success": "Правильно!",
                },
                {
                    "action": f"{a1}+{b1}",
                    "question": f"Сложим единицы: {a1} + {b1}. Сколько получится?",
                    "answer": str(sum1),
                    "hint": "Теперь складываем только единицы.",
                    "success": "Да, единицы посчитаны!",
                },
                {
                    "action": f"{sum10}+{sum1}",
                    "question": f"Осталось сложить {sum10} и {sum1}. Какой итог?",
                    "answer": str(result),
                    "hint": "Это последний шаг.",
                    "success": "Супер! Пример решён.",
                },
            ],
        }

    try:
        result = _safe_eval_math(expression)
        return {
            "task": task,
            "final_answer": _format_number(result),
            "full_solution": [f"{expression} = {_format_number(result)}"],
            "steps": [
                {
                    "action": expression,
                    "question": f"Посчитай: {expression}. Сколько получится?",
                    "answer": _format_number(result),
                    "hint": "Выполни действия по порядку.",
                    "success": "Верно!",
                }
            ],
        }
    except Exception:
        return {"steps": []}


# =========================
# 🧠 TEXT WORD PROBLEMS
# =========================
def _safe_json_loads(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_generated_data(data: dict, task: str):
    steps = data.get("steps") or []
    fixed_steps = []

    for step in steps:
        if not isinstance(step, dict):
            continue

        question = str(step.get("question", "")).strip()
        answer = str(step.get("answer", "")).strip()
        action = str(step.get("action", "")).strip()
        hint = str(step.get("hint", "Подумай, какое действие подходит.")).strip()
        success = str(step.get("success", "Верно! Идём дальше.")).strip()

        if not question or not answer:
            continue

        fixed_steps.append(
            {
                "action": action,
                "question": question,
                "answer": answer,
                "hint": hint,
                "success": success,
            }
        )

    final_answer = str(data.get("final_answer", "")).strip()
    full_solution = data.get("full_solution") or []

    if isinstance(full_solution, str):
        full_solution = [full_solution]

    full_solution = [str(item).strip() for item in full_solution if str(item).strip()]

    if not final_answer and fixed_steps:
        final_answer = fixed_steps[-1]["answer"]

    return {
        "task": task,
        "final_answer": final_answer,
        "full_solution": full_solution,
        "steps": fixed_steps,
    }


def generate_word_problem_steps(task: str):
    """
    Сначала модель решает текстовую задачу внутри себя,
    затем возвращает JSON: скрытое полное решение + шаги для диалога.
    В ChatScreen показываются только question/hint, а answer нужен для проверки.
    """
    prompt = f"""
Ты математический наставник для ученика 4 класса.

Нужно решить текстовую задачу, но ребёнку в чате НЕ выдавать готовое решение сразу.
Сделай так, чтобы приложение вело ребёнка по шагам наводящими вопросами.

ЗАДАЧА:
{task}

Верни строго JSON без markdown:
{{
  "final_answer": "только итоговое число или короткий ответ",
  "full_solution": [
    "Шаг 1: полное объяснение для скрытого экрана решения",
    "Шаг 2: полное объяснение для скрытого экрана решения"
  ],
  "steps": [
    {{
      "action": "математическое действие, например 176+234",
      "question": "наводящий вопрос ребёнку, без готового ответа",
      "answer": "короткий правильный ответ для проверки, лучше число",
      "hint": "подсказка без готового ответа",
      "success": "короткая похвала после правильного ответа"
    }}
  ]
}}

Правила:
- Для каждого важного действия создай отдельный шаг.
- Не задавай вопросы вида "что известно?" и "что нужно найти?".
- Каждый question должен вести к конкретному вычислению или конкретному выводу.
- В question и hint нельзя раскрывать правильный answer.
- В answer пиши только то, что ребёнок должен ввести: например "410", "586", "1172".
- Если задача про "на 234 больше", используй сложение.
- Если задача про "вместе" или "всего", используй сложение.
- Не добавляй числа, которых нельзя получить из условий и предыдущих шагов.
- Язык простой, дружелюбный, для 4 класса.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )

    raw = res.choices[0].message.content or "{}"
    data = _safe_json_loads(raw)
    data = _normalize_generated_data(data, task)

    if not data["steps"]:
        raise ValueError("Model returned no steps")

    return data


# =========================
# 📸 OCR
# =========================
@app.post("/api/vision")
async def vision(file: UploadFile = File(...)):
    img = await file.read()
    b64 = base64.b64encode(img).decode()

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Распознай текст на изображении. Верни только текст задачи без комментариев.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
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
        if not task:
            raise ValueError("Empty task")

        if is_simple_expression(task):
            simple = generate_simple_steps(task)
            if simple.get("steps"):
                return simple

        return generate_word_problem_steps(task)

    except Exception as e:
        print("/api/generate error:", repr(e))
        return {
            "task": task,
            "final_answer": "",
            "full_solution": [],
            "steps": [
                {
                    "action": "",
                    "question": "Не смог разобрать задачу 😢 Попробуй отправить её ещё раз или чуть проще.",
                    "answer": "",
                    "hint": "Проверь, что в задаче видны все числа и вопрос.",
                    "success": "",
                }
            ],
        }


# =========================
# ✅ CHECK ANSWER
# =========================
def _extract_math_part(text: str):
    text = text.lower().replace(",", ".")
    text = text.replace("×", "*").replace("x", "*").replace("х", "*").replace("÷", "/")
    text = re.sub(r"[^0-9\+\-\*/\(\)\.]", "", text)
    return text.strip()


def _answer_to_value(text: str):
    expr = _extract_math_part(text)
    if not expr:
        return None

    try:
        return _safe_eval_math(expr)
    except Exception:
        return expr


def _values_equal(a: Any, b: Any):
    if a is None or b is None:
        return False

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 0.000001

    return str(a).replace(" ", "") == str(b).replace(" ", "")


@app.post("/api/check_answer")
async def check_answer(req: CheckAnswerRequest):
    try:
        user_value = _answer_to_value(req.user_answer)
        correct_value = _answer_to_value(req.correct_answer)

        return {"correct": _values_equal(user_value, correct_value)}

    except Exception:
        return {"correct": False}
