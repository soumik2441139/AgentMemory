import os
from openai import OpenAI
from memory_v1_0 import AgentMemory
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def chat(session_id="default"):
    memory = AgentMemory(session_id)
    print(f"🧠 AgentMemory v1.0 | Session: {session_id}")
    print("Commands: 'quit' | 'stats' | 'sessions'\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'stats':
            import json
            print(json.dumps(memory.stats(), indent=2))
            continue
        if user_input.lower() == 'sessions':
            print(memory.db.list_sessions())
            continue
        if not user_input:
            continue

        memory.add_message("user", user_input)
        context = memory.get_context(user_input)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=context if context else [
                {"role": "user", "content": user_input}
            ]
        )

        reply = response.choices[0].message.content
        memory.add_message("assistant", reply)
        print(f"Agent: {reply}\n")

if __name__ == "__main__":
    chat("soumik-v1")
