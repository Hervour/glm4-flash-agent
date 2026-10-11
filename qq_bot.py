import asyncio
import json
import uuid
from typing import Any

import websockets
from websockets.server import WebSocketServerProtocol

from agent_core import Agent, create_client, load_env

QQ_SYSTEM_PROMPT = (
    "你是一个 QQ 智能助手，可以通过工具完成计算、查询时间和读取本地文件等任务。"
    "请根据用户需求决定是否调用工具；若不需要工具，直接回答即可。"
    "回答请使用中文，简洁自然，适合在 QQ 聊天中阅读，避免过长。"
)


def env_bool(value: str, default: bool = False) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_group_list(value: str) -> set[str]:
    if not value.strip():
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def extract_plain_text(message: Any) -> str:
    if isinstance(message, str):
        return message.strip()
    if isinstance(message, list):
        parts: list[str] = []
        for segment in message:
            if not isinstance(segment, dict):
                continue
            if segment.get("type") == "text":
                parts.append(segment.get("data", {}).get("text", ""))
            elif segment.get("type") == "at":
                parts.append(f"@{segment.get('data', {}).get('qq', '')}")
        return "".join(parts).strip()
    return ""


class QQBot:
    def __init__(self):
        self.env = load_env()
        self.host = self.env.get("QQ_WS_HOST", "127.0.0.1")
        self.port = int(self.env.get("QQ_WS_PORT", "8080"))
        self.access_token = self.env.get("QQ_ACCESS_TOKEN", "")
        self.reply_private = env_bool(self.env.get("QQ_REPLY_PRIVATE", "true"), True)
        self.reply_groups = parse_group_list(self.env.get("QQ_REPLY_GROUPS", ""))
        self.group_need_at = env_bool(self.env.get("QQ_GROUP_NEED_AT", "true"), True)

        client, model = create_client()
        self.client = client
        self.model = model
        self.agents: dict[str, Agent] = {}
        self.session_locks: dict[str, asyncio.Lock] = {}
        self.connections: set[WebSocketServerProtocol] = set()

    def _get_session_id(self, event: dict[str, Any]) -> str:
        message_type = event.get("message_type", "unknown")
        user_id = event.get("user_id")
        if message_type == "group":
            return f"group:{event.get('group_id')}:{user_id}"
        return f"private:{user_id}"

    def _get_agent(self, session_id: str) -> Agent:
        if session_id not in self.agents:
            self.agents[session_id] = Agent(self.client, self.model, QQ_SYSTEM_PROMPT)
            self.session_locks[session_id] = asyncio.Lock()
        return self.agents[session_id]

    def _should_reply(self, event: dict[str, Any], text: str) -> bool:
        if event.get("post_type") != "message":
            return False
        if event.get("sub_type") not in (None, "friend", "normal"):
            return False
        if not text:
            return False

        self_id = str(event.get("self_id", ""))
        user_id = str(event.get("user_id", ""))
        if self_id and user_id == self_id:
            return False

        message_type = event.get("message_type")
        if message_type == "private":
            return self.reply_private

        if message_type == "group":
            group_id = str(event.get("group_id", ""))
            if self.reply_groups and group_id not in self.reply_groups:
                return False
            if self.group_need_at:
                return self._message_mentions_self(event)
            return True

        return False

    def _message_mentions_self(self, event: dict[str, Any]) -> bool:
        message = event.get("message")
        self_id = str(event.get("self_id", ""))
        if isinstance(message, list):
            for segment in message:
                if segment.get("type") == "at" and str(segment.get("data", {}).get("qq")) == self_id:
                    return True
        raw_message = event.get("raw_message", "")
        return f"@{self_id}" in raw_message

    async def _call_api(
        self,
        websocket: WebSocketServerProtocol,
        action: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        echo = str(uuid.uuid4())
        payload = {"action": action, "params": params, "echo": echo}
        await websocket.send(json.dumps(payload, ensure_ascii=False))

        while True:
            raw = await websocket.recv()
            data = json.loads(raw)
            if data.get("echo") == echo:
                return data
            await self._handle_event(websocket, data)

    async def _send_reply(self, websocket: WebSocketServerProtocol, event: dict[str, Any], text: str):
        if event.get("message_type") == "group":
            action = "send_group_msg"
            params = {"group_id": event["group_id"], "message": text}
        else:
            action = "send_private_msg"
            params = {"user_id": event["user_id"], "message": text}

        result = await self._call_api(websocket, action, params)
        if result.get("status") != "ok":
            print(f"[QQ] 发送失败: {result}")

    async def _handle_message_event(self, websocket: WebSocketServerProtocol, event: dict[str, Any]):
        text = extract_plain_text(event.get("message"))
        if not self._should_reply(event, text):
            return

        session_id = self._get_session_id(event)
        agent = self._get_agent(session_id)
        lock = self.session_locks[session_id]

        sender = event.get("sender", {})
        nickname = sender.get("card") or sender.get("nickname") or str(event.get("user_id"))
        print(f"[QQ] 收到消息 ({session_id}) {nickname}: {text}")

        async with lock:
            loop = asyncio.get_running_loop()
            try:
                reply = await loop.run_in_executor(None, lambda: agent.chat(text, verbose=False))
            except Exception as exc:
                reply = f"处理失败：{exc}"
                print(f"[QQ] Agent 出错: {exc}")

        if reply:
            await self._send_reply(websocket, event, reply)
            print(f"[QQ] 已回复 ({session_id}): {reply[:80]}{'...' if len(reply) > 80 else ''}")

    async def _handle_event(self, websocket: WebSocketServerProtocol, data: dict[str, Any]):
        if data.get("post_type") == "message":
            await self._handle_message_event(websocket, data)
        elif data.get("meta_event_type") == "lifecycle":
            print("[QQ] NapCat 已连接")
        elif data.get("meta_event_type") == "heartbeat":
            pass

    async def _connection_handler(self, websocket: WebSocketServerProtocol):
        if self.access_token:
            headers = websocket.request.headers if websocket.request else {}
            auth_header = headers.get("Authorization", "")
            expected = f"Bearer {self.access_token}"
            if auth_header != expected:
                print("[QQ] 连接被拒绝：Access Token 不匹配")
                await websocket.close(code=1008, reason="Unauthorized")
                return

        self.connections.add(websocket)
        print(f"[QQ] 新连接: {websocket.remote_address}")
        try:
            async for raw in websocket:
                data = json.loads(raw)
                await self._handle_event(websocket, data)
        except websockets.ConnectionClosed:
            print("[QQ] 连接已断开")
        finally:
            self.connections.discard(websocket)

    async def run(self):
        print("=" * 50)
        print("  QQ Agent Bot")
        print(f"  监听地址: ws://{self.host}:{self.port}/")
        print("  请先在 NapCat 中配置反向 WebSocket 连接到此地址")
        print("=" * 50)
        print(f"  私聊回复: {'开启' if self.reply_private else '关闭'}")
        if self.reply_groups:
            print(f"  群聊回复: {', '.join(sorted(self.reply_groups))}")
        else:
            print("  群聊回复: 关闭（可在 api.env 配置 QQ_REPLY_GROUPS）")
        print(f"  群聊需 @: {'是' if self.group_need_at else '否'}")
        print("=" * 50)

        async with websockets.serve(self._connection_handler, self.host, self.port):
            await asyncio.Future()


def main():
    try:
        bot = QQBot()
    except ValueError as exc:
        print(f"配置错误：{exc}")
        return

    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\nQQ Bot 已停止")


if __name__ == "__main__":
    main()
