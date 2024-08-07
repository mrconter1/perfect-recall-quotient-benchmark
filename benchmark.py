import re
import asyncio
from data import QUOTES_AND_TITLES
from api_provider import OpenRouterProvider
from tabulate import tabulate
from tqdm import tqdm

def create_prompt(quote):
    return f"""You have been exposed to a corpus of scientific papers. Your task is to demonstrate perfect recall by providing the exact title of the paper containing the following quote:

"{quote}"

Please respond using the following format:
<TITLE>
[Your answer here]
</TITLE>

If you cannot recall the exact title with certainty, respond with:
<TITLE>
Unable to recall
</TITLE>

Your response will be evaluated for accuracy and used to assess your recall capabilities. Ensure your entire response is contained within these tags."""

async def process_question(provider, model_name, title, quote):
    if not quote:
        return "Blank"
    
    prompt = create_prompt(quote)
    try:
        response = await provider.send_prompt(prompt, model_name)
        
        match = re.search(r'<TITLE>\s*(.*?)\s*</TITLE>', response, re.DOTALL | re.IGNORECASE)
        if match:
            extracted_title = match.group(1).strip()
            if extracted_title.lower() == title.strip().lower():
                print(f"\nCorrect answer for {model_name}:")
                print(f"Quote: \"{quote}\"")
                print(f"Entire response:\n{response}")
                return "Correct"
            elif extracted_title.lower() == "unable to recall":
                return "Unable to recall"
            else:
                return "Incorrect"
        else:
            return "Parse error"
    except Exception as e:
        return "Error"

async def process_model(model_name, papers, pbar):
    provider = OpenRouterProvider()
    correct_answers = 0
    total_questions = 0
    results = []

    for paper in papers:
        for quote in paper['quotes']:
            result = await process_question(provider, model_name, paper['title'], quote)
            if result == "Correct":
                correct_answers += 1
            if result != "Blank":
                total_questions += 1
            results.append(result)
            pbar.update(1)

    ptrq_score = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    display_name = model_name.split('/')[-1]
    print(f"\n{display_name} completed. Score: {ptrq_score:.2f}% ({correct_answers}/{total_questions})")
    return [display_name, f"{ptrq_score:.2f}", f"{correct_answers}/{total_questions}"]

async def run_benchmark(models):
    total_tasks = len(models) * sum(len(paper['quotes']) for paper in QUOTES_AND_TITLES)
    results = []

    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        tasks = [process_model(model, QUOTES_AND_TITLES, pbar) for model in models]
        results = await asyncio.gather(*tasks)

    return results

def print_results_table(results):
    headers = ["Model", "PTRQ Score", "Correct/Total"]
    print("\nFinal Results:")
    print(tabulate(results, headers=headers, tablefmt="grid"))