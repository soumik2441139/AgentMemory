import os
import json
from openai import OpenAI
from agentmemory import AgentMemory
from dotenv import load_dotenv

load_dotenv()


def get_client_and_model():
    if os.getenv("OPENAI_API_KEY"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        provider = f"OpenAI ({model})"
    elif os.getenv("GROQ_API_KEY"):
        client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        provider = f"Groq ({model})"
    elif os.getenv("DEEPSEEK_API_KEY"):
        client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        provider = f"DeepSeek ({model})"
    else:
        raise ValueError("No API key found! Please set OPENAI_API_KEY, GROQ_API_KEY, or DEEPSEEK_API_KEY in your .env file.")

    return client, model, provider


def _auto_summarize_enabled(default=True):
    value = os.getenv("AGENTMEMORY_AUTO_SUMMARIZE")
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def chat(session_id="default", auto_summarize=None):
    client, model, provider = get_client_and_model()
    if auto_summarize is None:
        auto_summarize = _auto_summarize_enabled(default=True)
    memory = AgentMemory(session_id, auto_summarize=auto_summarize)
    print(f"AgentMemory | Provider: {provider} | Session: {session_id} | Auto-summarize: {auto_summarize}")
    print("Commands: 'quit' | 'stats'\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            print(json.dumps(memory.stats(), indent=2))
            continue
        if not user_input:
            continue

        memory.add_message("user", user_input)
        context = memory.get_context(user_input)

        # Prepare messages ensuring the current user prompt is present
        messages = list(context) if context else []
        if not messages or messages[-1]["content"] != user_input:
            messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=model,
            messages=messages
        )

        reply = response.choices[0].message.content
        memory.add_message("assistant", reply)
        print(f"Agent: {reply}\n")


if __name__ == "__main__":
    chat()

