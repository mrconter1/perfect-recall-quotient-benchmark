import json

def load_quotes_and_titles():
    with open('data.json', 'r') as file:
        return json.load(file)

QUOTES_AND_TITLES = load_quotes_and_titles()