import os
from openai import OpenAI
from memory import AgentMemory
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def chat(session_id="default"):
    memory = AgentMemory(session_id, hot_limit=10)
    print(f"🧠 AgentMemory started | Session: {session_id}")
    print("Type 'quit' to exit | Type 'stats' to see memory stats\n")

    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        if user_input.lower() == 'stats':
            print(f"📊 {memory.stats()}\n")
            continue
        if not user_input:
            continue

        # Add user message to memory
        memory.add_message("user", user_input)

        # Get context from memory
        context = memory.get_context()

        # Call DeepSeek with memory context
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=context
        )

        reply = response.choices[0].message.content

        # Add assistant reply to memory
        memory.add_message("assistant", reply)

        print(f"Agent: {reply}\n")
        print(f"[Memory: {memory.stats()['hot_messages']} hot | {memory.stats()['cold_messages']} cold]\n")

if __name__ == "__main__":
    chat("soumik-session-1")
