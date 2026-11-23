"""
Evaluation runner comparing context retrieval strategies on long multi-turn conversations.
Measures Recall Accuracy, Estimated Tokens Used, and Number of Context Messages.
"""

from generate_conversations import SCENARIOS, build_conversation
from strategies import (
    LastNStrategy,
    KeywordChunkStrategy,
    TFIDFChunkStrategy,
    DenseEmbeddingStrategy
)

def estimate_tokens(messages):
    text = " ".join([m["content"] for m in messages])
    # Approximate 1 token ~= 4 characters / 0.75 words
    return int(len(text.split()) * 1.3)


def contains_all_keywords(text, keywords):
    text_lower = text.lower()
    return all(k.lower() in text_lower for k in keywords)


def run_benchmark():
    strategies = {
        "Last-10 Messages": LastNStrategy(10),
        "Last-20 Messages": LastNStrategy(20),
        "Keyword Chunking": KeywordChunkStrategy(chunk_size=6),
        "AgentMemory TF-IDF": TFIDFChunkStrategy(),
        "Char N-Gram TF-IDF": DenseEmbeddingStrategy(),
    }

    results = {name: {"hits": 0, "total": 0, "tokens": 0, "messages": 0} for name in strategies}

    print("=" * 72)
    print("🧠 AGENTMEMORY EVALUATION HARNESS — BENCHMARK RESULTS")
    print("=" * 72)

    for scenario in SCENARIOS:
        convo = build_conversation(scenario, padding_multiplier=3)
        total_turns = len(convo)
        print(f"\nScenario: '{scenario['id']}' ({total_turns} turns total, fact at turn 3)")
        print(f"Query: \"{scenario['query']}\"")
        print(f"Target Keywords: {scenario['expected_answer_keywords']}")

        for name, strat in strategies.items():
            retrieved = strat.retrieve(convo, scenario["query"])
            retrieved_text = " ".join([m["content"] for m in retrieved])
            hit = contains_all_keywords(retrieved_text, scenario["expected_answer_keywords"])
            tokens = estimate_tokens(retrieved)

            results[name]["total"] += 1
            if hit:
                results[name]["hits"] += 1
            results[name]["tokens"] += tokens
            results[name]["messages"] += len(retrieved)

            status = "✅ HIT " if hit else "❌ MISS"
            print(f"  {status} | {name:<22} | msgs: {len(retrieved):<2} | ~tokens: {tokens:<4}")

    # Summary Table
    print("\n" + "=" * 72)
    print(f"{'Strategy':<24} | {'Recall':<8} | {'Avg Tokens':<10} | {'Avg Context Msgs'}")
    print("-" * 72)

    for name, res in results.items():
        recall = (res["hits"] / res["total"]) * 100
        avg_tokens = int(res["tokens"] / res["total"])
        avg_msgs = round(res["messages"] / res["total"], 1)
        print(f"{name:<24} | {recall:>5.1f}%   | {avg_tokens:>8}   | {avg_msgs:>8}")

    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
