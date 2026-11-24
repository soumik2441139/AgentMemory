"""
Benchmark dataset generator for long conversation context retrieval evaluation.
Creates scenarios where an early fact is embedded among multiple distracting topics.
"""

SCENARIOS = [
    {
        "id": "dog_name",
        "fact": "My golden retriever is named Barnaby and he loves peanut butter.",
        "query": "What is my dog's name and favorite treat?",
        "expected_answer_keywords": ["barnaby", "peanut butter"],
        "fact_topic": "Pets",
        "distractor_topics": [
            ("Weather", [
                "It has been raining continuously for three days here.",
                "I hope tomorrow is sunny so I can go hiking.",
                "The meteorological department forecast a thunderstorm over the weekend.",
                "Do you think climate change is causing more frequent storms?"
            ]),
            ("Programming", [
                "I am refactoring a legacy Django web service into FastAPI.",
                "How does async and await work in Python event loops?",
                "Pydantic v2 has incredible serialization performance improvements.",
                "PostgreSQL indexing with pgvector is great for vector search."
            ]),
            ("Cooking", [
                "I made sourdough bread using active starter yeast yesterday.",
                "The hydration ratio was roughly 75 percent with whole wheat flour.",
                "Baking at 450 degrees in a Dutch oven produces wonderful crust blister."
            ]),
            ("Travel", [
                "I am planning a trip to Kyoto and Tokyo in November.",
                "I want to ride the Shinkansen bullet train from Osaka.",
                "Autumn leaves in Arashiyama bamboo forest are supposed to be stunning."
            ])
        ]
    },
    {
        "id": "database_port",
        "fact": "Our staging Redis cluster is listening on custom port 16379 with TLS enabled.",
        "query": "Which port does our staging Redis cluster run on?",
        "expected_answer_keywords": ["16379"],
        "fact_topic": "Infrastructure",
        "distractor_topics": [
            ("Coffee", [
                "I recently bought a manual lever espresso machine.",
                "Grinding beans with a conical burr grinder reduces fines.",
                "A 1 to 2 brew ratio in 28 seconds creates rich crema."
            ]),
            ("History", [
                "The Apollo 11 moon landing occurred in July 1969.",
                "The guidance computer had only four kilobytes of RAM.",
                "Margaret Hamilton led the software engineering team at MIT."
            ]),
            ("Music", [
                "I have been listening to classical piano concertos by Rachmaninoff.",
                "The second piano concerto in C minor has a gorgeous melodic theme.",
                "Learning to play Chopin nocturnes requires great finger dexterity."
            ])
        ]
    },
    {
        "id": "favorite_editor",
        "fact": "My primary code editor is Neovim configured with Lua and telescope.",
        "query": "What code editor and plugin do I use?",
        "expected_answer_keywords": ["neovim", "telescope"],
        "fact_topic": "Dev Tools",
        "distractor_topics": [
            ("Gardening", [
                "I planted cherry tomato seedlings in the greenhouse.",
                "They require at least six hours of direct sunlight each day.",
                "Adding organic compost improved the soil drainage significantly."
            ]),
            ("Hardware", [
                "The new M-series chips have unified memory architecture.",
                "Memory bandwidth reaches hundreds of gigabytes per second.",
                "Fan noise is almost imperceptible under high compiler workloads."
            ]),
            ("Fitness", [
                "I completed a 5k running session this morning in 24 minutes.",
                "Proper footwear with arch support prevents shin splints.",
                "Hydration and electrolytes are essential after endurance workouts."
            ])
        ]
    }
]

# These queries deliberately have no relevant topic in their conversation.
# They measure retrieval precision separately from the fact-recall scenarios.
NO_MATCH_CASES = [
    {
        "id": "quantum_physics",
        "messages": [
            {"role": "user", "content": "Our Redis cache uses TLS and runs on port 16379."},
            {"role": "assistant", "content": "Noted the Redis deployment details."},
            {"role": "user", "content": "Transformer attention helps language models focus on tokens."},
            {"role": "assistant", "content": "That is a useful machine learning observation."},
        ],
        "query": "Explain quantum entanglement in particle physics.",
    },
    {
        "id": "bread_baking",
        "messages": [
            {"role": "user", "content": "My primary code editor is Neovim with Lua and telescope."},
            {"role": "assistant", "content": "Noted your development environment."},
            {"role": "user", "content": "Cherry tomato seedlings need sunlight and well drained soil."},
            {"role": "assistant", "content": "That should help the garden thrive."},
        ],
        "query": "How should I feed a sourdough starter before baking bread?",
    },
    {
        "id": "kubernetes_scheduling",
        "messages": [
            {"role": "user", "content": "I enjoy classical piano concertos by Rachmaninoff."},
            {"role": "assistant", "content": "Those concertos have memorable melodies."},
            {"role": "user", "content": "A manual espresso machine can produce rich crema."},
            {"role": "assistant", "content": "Grind size and brew ratio matter too."},
        ],
        "query": "How does Kubernetes schedule pods across cluster nodes?",
    },
]

# No-match queries asked at the end of the long fact-recall conversations.
# Unlike NO_MATCH_CASES, these run against 10-16 noisy topic chunks. Passing
# means no topic-chunk message is returned (General Chat greetings are allowed).
LONG_NO_MATCH_CASES = [
    {"scenario_id": "dog_name", "query": "How do I replace a punctured bicycle tube?"},
    {"scenario_id": "database_port", "query": "What stretches relieve sciatica discomfort?"},
    {"scenario_id": "favorite_editor", "query": "How are employee stock options taxed?"},
]


def build_conversation(scenario, padding_multiplier=3):
    """
    Constructs a sequential list of messages:
    - Small talk greetings
    - Fact planted early (Turn 3)
    - Extensive distracting conversations from other topics
    """
    messages = [
        {"role": "user", "content": "Hello there! Good morning."},
        {"role": "assistant", "content": "Good morning! How can I help you today?"},
        {"role": "user", "content": scenario["fact"]},
        {"role": "assistant", "content": f"Understood, noted about {scenario['fact_topic']}."}
    ]

    for _ in range(padding_multiplier):
        for topic_name, turn_list in scenario["distractor_topics"]:
            for content in turn_list:
                messages.append({"role": "user", "content": content})
                messages.append({"role": "assistant", "content": f"Interesting point about {topic_name}."})

    return messages
