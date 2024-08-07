from data import QUOTES_AND_TITLES
from api_provider import OpenRouterProvider
from tabulate import tabulate

import re

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

async def run_benchmark(models):
    provider = OpenRouterProvider()
    results = []

    for model_name in models:
        correct_answers = 0
        total_questions = len(QUOTES_AND_TITLES)

        for item in QUOTES_AND_TITLES:
            prompt = create_prompt(item['quote'])
            try:
                response = await provider.send_prompt(prompt, model_name)
                
                # Extract the title from the response
                match = re.search(r'<TITLE>\s*(.*?)\s*</TITLE>', response, re.DOTALL | re.IGNORECASE)
                if match:
                    extracted_title = match.group(1).strip()
                    if extracted_title.lower() == item['title'].strip().lower():
                        correct_answers += 1
                    elif extracted_title.lower() == "unable to recall":
                        # Count this as an incorrect answer, but don't print a warning
                        pass
                    else:
                        print(f"Incorrect answer for model {model_name}. Expected: {item['title']}, Got: {extracted_title}")
                else:
                    print(f"Warning: Could not parse response for model {model_name}: {response}")
            except Exception as e:
                print(f"Error occurred while processing model {model_name}: {str(e)}")
                # You might want to decide how to handle this case. 
                # For now, we'll count it as an incorrect answer
                pass

        ptrq_score = (correct_answers / total_questions) * 100
        display_name = model_name.split('/')[-1]  # Get the text after the last '/'
        results.append([display_name, f"{ptrq_score:.2f}", f"{correct_answers}/{total_questions}"])

    return results

def print_results_table(results):
    headers = ["Model", "PTRQ Score", "Correct/Total"]
    print(tabulate(results, headers=headers, tablefmt="grid"))