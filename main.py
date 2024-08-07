import asyncio
from benchmark import run_benchmark, print_results_table

MODELS = [
    "google/gemini-pro-1.5",
    "anthropic/claude-3-opus",
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-3.5-turbo",
    "openai/gpt-4-turbo",
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-405b-instruct",
    "mistralai/mistral-large"
]

NUMBER_OF_ATTEMPTS = 5

async def main():
    results = await run_benchmark(MODELS, number_of_attempts=NUMBER_OF_ATTEMPTS)
    print_results_table(results)

if __name__ == "__main__":
    asyncio.run(main())