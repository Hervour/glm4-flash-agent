# GLM-4-Flash Agent

A small tool-using AI assistant based on ZhipuAI GLM-4-Flash. It supports local calculation, current time, bounded file reading, and a OneBot/NoneBot QQ WebSocket adapter.

## Setup

1. Copy `api.env.example` to `api.env`.
2. Fill in your ZhipuAI API key and QQ connection settings.
3. Install dependencies and run:

```powershell
python -m pip install -r requirements.txt
python main.py
```

For the QQ adapter, start a compatible OneBot WebSocket service and run `python qq_bot.py`.
