from setuptools import setup, find_packages

setup(
    name="llmeval",
    version="0.1.0",
    author="LLMEval Contributors",
    description="An open-source LLM output evaluation framework (faithfulness, relevance, hallucination, coherence)",
    long_description=open("README.md", "r", encoding="utf-8").read() if open("README.md", "r", encoding="utf-8") else "",
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "openai>=1.0.0",
        "anthropic>=0.3.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.22.0",
        "click>=8.0.0",
        "rich>=13.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llmeval=llmeval.cli:cli",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
