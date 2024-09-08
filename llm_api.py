import os
prompt = None
with open('./generic_prompt.txt') as f:
    prompt = f.readlines()


def create_open_ai_object():
    from openai import OpenAI
    client = OpenAI()
    return client


def create_gemini_object():
    import google.generativeai as genai
    genai.configure(api_key=os.environ["API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    return model
# create_open_ai_object()


def populate_dialogue_list(excel_path, dialogue_list):
    import pandas as pd
    dialogue_df = pd.read_excel('./test_dialogues.xlsx')


def add_theme_to_dialogue_list(theme_text, dialogue_list):


def format_llm_input(title, description, category, theme_prompt):
    init_dialogue = f"Title: {title},Description: {description},Category: {category},Theme:{theme_prompt}"
    final_prompt = str(prompt)+str(init_dialogue)
    return final_prompt


def dialogue_prompt_flow():
    '''
    step 1: import feedback form into singular text
    step 2: clean or add additional metadata to dialogue
    step 3: wrap the cleaned text with prompt (check token count limit)
    step 4: put cleaned processed dialogue into tag generator 1,2,3,4...n based on definition of prompt
    step 5: store tags of each category into single dicstionary object which is our dialogue 

    '''
