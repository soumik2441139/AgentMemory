import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

SUMMARIZE_THRESHOLD = 6  # summarize chunk when it hits this many messages


def _get_summarizer_client():
    if os.getenv("OPENAI_API_KEY"):
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY")), os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    elif os.getenv("GROQ_API_KEY"):
        return OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        ), os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    elif os.getenv("DEEPSEEK_API_KEY"):
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        ), "deepseek-chat"
    return None, None


def summarize_chunk(chunk):
    messages = chunk["messages"]
    
    if len(messages) < SUMMARIZE_THRESHOLD:
        return None  # not ready to summarize yet

    client, model = _get_summarizer_client()
    if not client:
        return None

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

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"  Summarization skipped: {e}")
        return None


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
    print(f"  Summarized chunk '{chunk['title']}' ({len(messages)} -> {len(chunk['messages'])} messages)")
    return chunk
