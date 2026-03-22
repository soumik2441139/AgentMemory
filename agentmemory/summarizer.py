import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

SUMMARIZE_THRESHOLD = 6  # summarize chunk when it hits this many messages

def summarize_chunk(chunk):
    messages = chunk["messages"]
    
    if len(messages) < SUMMARIZE_THRESHOLD:
        return None  # not ready to summarize yet

    # Build conversation text
    conversation = "\n".join([
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
    ])

    prompt = f"""Summarize this conversation in 2-3 sentences. 
Be concise. Capture the key topics and conclusions only.
Do not add any extra commentary.

CONVERSATION:
{conversation}

SUMMARY:"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150
    )

    summary = response.choices[0].message.content.strip()
    return summary


def maybe_summarize(chunk):
    messages = chunk["messages"]

    if len(messages) < SUMMARIZE_THRESHOLD:
        return chunk  # nothing to do

    summary = summarize_chunk(chunk)
    if not summary:
        return chunk

    # Replace all messages with one summary message
    # Keep only the last 2 messages for fresh context
    last_two = messages[-2:]

    chunk["messages"] = [
        {
            "role": "system",
            "content": f"[SUMMARY OF EARLIER CONVERSATION]: {summary}",
            "timestamp": messages[0]["timestamp"],
            "summarized": True
        }
    ] + last_two

    chunk["summarized"] = True
    print(f"  📝 Summarized chunk '{chunk['title']}' ({len(messages)} → {len(chunk['messages'])} messages)")
    return chunk
