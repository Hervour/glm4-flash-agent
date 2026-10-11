import ast
import json
import operator
import re
from datetime import datetime
from pathlib import Path

from zhipuai import ZhipuAI

ENV_PATH = Path(__file__).parent / "api.env"
SYSTEM_PROMPT = (
    "你是一个智能助手 Agent，可以通过工具完成计算、查询时间和读取本地文件等任务。"
    "请根据用户需求决定是否调用工具；若不需要工具，直接回答即可。"
    "回答请使用中文，简洁清晰。"
)

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def get_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> str:
    expression = expression.strip()
    if not re.fullmatch(r"[\d\s+\-*/().]+", expression):
        return "错误：表达式仅支持数字和 + - * / ( ) 运算符"
    try:
        node = ast.parse(expression, mode="eval").body
        result = _eval_node(node)
        return str(result)
    except (SyntaxError, TypeError, ZeroDivisionError, ValueError) as exc:
        return f"错误：{exc}"


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in SAFE_OPERATORS:
        return SAFE_OPERATORS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    raise ValueError("不支持的表达式")


def read_file(file_path: str) -> str:
    base = Path(__file__).parent.resolve()
    target = (base / file_path).resolve()
    if not str(target).startswith(str(base)):
        return "错误：不允许读取项目目录外的文件"
    if not target.exists():
        return f"错误：文件不存在 - {file_path}"
    if not target.is_file():
        return f"错误：路径不是文件 - {file_path}"
    content = target.read_text(encoding="utf-8")
    if len(content) > 4000:
        return content[:4000] + "\n...(内容已截断)"
    return content


TOOL_FUNCTIONS = {
    "get_current_time": lambda _: get_current_time(),
    "calculate": lambda args: calculate(args["expression"]),
    "read_file": lambda args: read_file(args["file_path"]),
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式，支持 + - * / 和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如 2+3*4",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "读取项目目录内的文本文件内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "相对于项目根目录的文件路径，例如 main.py",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
]


class Agent:
    def __init__(self, client: ZhipuAI, model: str, system_prompt: str = SYSTEM_PROMPT):
        self.client = client
        self.model = model
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]

    def _call_model(self):
        return self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOLS,
            tool_choice="auto",
        )

    def _run_tool(self, name: str, arguments: str) -> str:
        if name not in TOOL_FUNCTIONS:
            return f"错误：未知工具 {name}"
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            return "错误：工具参数不是合法 JSON"
        return TOOL_FUNCTIONS[name](args)

    def chat(self, user_input: str, verbose: bool = True) -> str:
        self.messages.append({"role": "user", "content": user_input})

        while True:
            response = self._call_model()
            message = response.choices[0].message

            assistant_message = message.model_dump(exclude_none=True)
            self.messages.append(assistant_message)

            tool_calls = message.tool_calls or []
            if not tool_calls:
                return message.content or ""

            for tool_call in tool_calls:
                result = self._run_tool(
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
                if verbose:
                    print(f"  [工具] {tool_call.function.name}({tool_call.function.arguments})")
                    print(f"  [结果] {result}")
                self.messages.append(
                    {
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call.id,
                    }
                )


def create_client() -> tuple[ZhipuAI, str]:
    env = load_env()
    api_key = env.get("API_KEY", "")
    model = env.get("MODEL", "glm-4-flash")

    if not api_key or api_key == "your_api_key_here":
        raise ValueError("请先在 api.env 中配置有效的 API_KEY")

    return ZhipuAI(api_key=api_key), model
