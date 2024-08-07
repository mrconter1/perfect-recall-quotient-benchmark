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

async def process_question(provider, model_name, title, quote, number_of_attempts, pbar):
    if not quote:
        pbar.update(number_of_attempts)  # Update progress bar for blank quotes
        return "Blank"
    
    for attempt in range(number_of_attempts):
        prompt = create_prompt(quote)
        try:
            response = await provider.send_prompt(prompt, model_name)
            
            match = re.search(r'<TITLE>\s*(.*?)\s*</TITLE>', response, re.DOTALL | re.IGNORECASE)
            if match:
                extracted_title = match.group(1).strip()
                if extracted_title.lower() == title.strip().lower():
                    print(f"\nCorrect answer for {model_name} (Attempt {attempt + 1}/{number_of_attempts}):")
                    print(f"Quote: \"{quote}\"")
                    print(f"Entire response:\n{response}")
                    pbar.update(number_of_attempts - attempt)  # Update progress for remaining attempts
                    return "Correct"
                elif extracted_title.lower() == "unable to recall":
                    if attempt == number_of_attempts - 1:
                        pbar.update(1)  # Update progress for last attempt
                        return "Unable to recall"
                else:
                    if attempt == number_of_attempts - 1:
                        pbar.update(1)  # Update progress for last attempt
                        return "Incorrect"
            else:
                if attempt == number_of_attempts - 1:
                    pbar.update(1)  # Update progress for last attempt
                    return "Parse error"
        except Exception as e:
            if attempt == number_of_attempts - 1:
                pbar.update(1)  # Update progress for last attempt
                return "Error"
        
        pbar.update(1)  # Update progress after each attempt
    
    return "Incorrect"  # If all attempts fail

async def process_model(model_name, papers, pbar, number_of_attempts):
    provider = OpenRouterProvider()
    correct_answers = 0
    total_questions = 0
    results = []

    for paper in papers:
        for quote in paper['quotes']:
            result = await process_question(provider, model_name, paper['title'], quote, number_of_attempts, pbar)
            if result == "Correct":
                correct_answers += 1
            if result != "Blank":
                total_questions += 1
            results.append(result)

    ptrq_score = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    display_name = model_name.split('/')[-1]
    print(f"\n{display_name} completed. Score: {ptrq_score:.2f}% ({correct_answers}/{total_questions})")
    return [display_name, f"{ptrq_score:.2f}", f"{correct_answers}/{total_questions}"]

async def run_benchmark(models, number_of_attempts):
    total_tasks = len(models) * sum(len(paper['quotes']) for paper in QUOTES_AND_TITLES) * number_of_attempts
    results = []

    with tqdm(total=total_tasks, desc="Overall Progress") as pbar:
        tasks = [process_model(model, QUOTES_AND_TITLES, pbar, number_of_attempts) for model in models]
        results = await asyncio.gather(*tasks)

    return results

def print_results_table(results):
    headers = ["Model", "PTRQ Score", "Correct/Total"]
    print("\nFinal Results:")
    print(tabulate(results, headers=headers, tablefmt="grid"))