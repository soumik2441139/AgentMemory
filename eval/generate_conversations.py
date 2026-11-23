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
