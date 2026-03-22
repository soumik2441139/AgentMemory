from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="agentmemory",
    version="1.1.0",
    author="Soumik",
    description="Lightweight AI agent memory management using Redis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/soumik2441139/AgentMemory",
    packages=find_packages(),
    install_requires=[
        "redis>=4.0.0",
        "openai>=1.0.0",
        "python-dotenv>=0.19.0",
        "nltk>=3.7",
        "scikit-learn>=1.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="ai agent memory redis llm context management",
)
