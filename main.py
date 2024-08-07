import asyncio
from benchmark import run_benchmark, print_results_table

MODELS = [
    "anthropic/claude-3.5-sonnet",
]

async def main():
    results = await run_benchmark(MODELS)
    print_results_table(results)

if __name__ == "__main__":
    asyncio.run(main())