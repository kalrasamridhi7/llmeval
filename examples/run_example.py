#!/usr/bin/env python3
"""
LLMEval Example Usage Script.

Demonstrates how to use the LLMEval Python API to evaluate a single output
and run batch evaluations using the sample dataset.
"""

import json
import os
import sys
from dotenv import load_dotenv

# Ensure the root of the project is in the Python path for import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llmeval import evaluate, evaluate_batch


def run_single_demo():
    print("=== 1. Evaluating Single Example ===")
    
    # We'll use a mocked-like context or real API call if keys are set
    question = "How do I initialize a new git repository?"
    context = (
        "To start tracking a new project with Git, open your terminal, "
        "navigate to your project directory, and run 'git init'. This creates a hidden "
        ".git directory that stores the repository metadata."
    )
    answer = "Open the terminal in your project directory and run 'git init' to initialize the repository."
    
    print(f"Question: {question}")
    print(f"Context: {context}")
    print(f"Answer: {answer}")
    
    # Run evaluation for Faithfulness and Relevance
    metrics = ["faithfulness", "relevance"]
    print(f"\nRunning evaluation on metrics: {metrics}...")
    
    try:
        results = evaluate(
            question=question,
            context=context,
            answer=answer,
            metrics=metrics,
            model="gpt-4o-mini"
        )
        print("\nResults:")
        print(json.dumps(results, indent=2))
    except Exception as e:
        print(f"\nEvaluation failed (Is your API key set?): {e}")


def run_batch_demo():
    print("\n=== 2. Evaluating Batch of Examples ===")
    
    sample_file = os.path.join(os.path.dirname(__file__), "sample_rag_output.json")
    if not os.path.exists(sample_file):
        print(f"Error: Sample file '{sample_file}' not found.")
        return
        
    print(f"Reading sample inputs from: {sample_file}")
    with open(sample_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Run evaluation on first 2 examples for speed in this demo
    subset = data[:2]
    print(f"Evaluating {len(subset)} examples across all metrics...")
    
    try:
        results = evaluate_batch(
            rows=subset,
            metrics=["all"],
            model="gpt-4o-mini"
        )
        print("\nBatch Results summary:")
        for idx, res in enumerate(results):
            print(f"\nExample #{idx+1}:")
            print(f"  Question: {res['input']['question']}")
            print(f"  Overall Score: {res['overall']:.2f}")
            print("  Per-metric scores:")
            for metric, score_details in res["scores"].items():
                print(f"    - {metric}: {score_details['score']:.2f} ({score_details['explanation'][:80]}...)")
    except Exception as e:
        print(f"\nBatch evaluation failed (Is your API key set?): {e}")


def main():
    # Load environment variables
    load_dotenv()
    
    # Check if API keys are set
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not openai_key and not anthropic_key:
        print("WARNING: Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY environment variables are set.")
        print("Real evaluation requests will fail. Please set them in your environment or a .env file.")
        print("----------------------------------------------------------------------------------------\n")
        
    run_single_demo()
    run_batch_demo()


if __name__ == "__main__":
    main()
