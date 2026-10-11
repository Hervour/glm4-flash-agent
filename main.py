from agent_core import Agent, create_client


def main():
    print("=" * 50)
    print("  GLM-4-Flash Agent")
    print("  输入问题开始对话，输入 quit 或 exit 退出")
    print("=" * 50)

    try:
        client, model = create_client()
    except ValueError as exc:
        print(f"配置错误：{exc}")
        return

    agent = Agent(client, model)

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("再见！")
            break

        try:
            reply = agent.chat(user_input)
            print(f"\nAgent: {reply}")
        except Exception as exc:
            print(f"\n请求失败：{exc}")


if __name__ == "__main__":
    main()
