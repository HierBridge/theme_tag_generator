import os
import time
import pandas as pd
import json
from typing import List, Dict, Any
from abc import ABC, abstractmethod
import re


class LLMClient(ABC):
    @abstractmethod
    def generate_response(self, prompt: str) -> str:
        pass


class OpenAIClient(LLMClient):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(
            api_key="")

    def generate_response(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


class GeminiClient(LLMClient):
    def __init__(self):
        import google.generativeai as genai
        genai.configure(api_key="")
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_response(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return response.text


class MetaTagGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt_template()
        self.themes: Dict[str, str] = {}
        self._load_predefined_themes()

    def _load_prompt_template(self) -> str:
        with open('./generic_prompt.txt', 'r') as f:
            return f.read()

    def _load_predefined_themes(self):
        predefined_themes = {
            "employee_individual": "Find or suggest closely related words for individual employee performance, skills, and personal development.",
            "team_performance": "Find or suggest closely related words for team dynamics, collaboration, and collective achievements.",
            "organizational_commitment": "Find or suggest closely related words for employee loyalty, dedication, and alignment with company values.",
            "organizational_citizenship_behavior": "Find or suggest closely related words for extra-role behaviors, volunteerism, and actions that benefit the organization beyond job requirements."
        }
        for theme, prompt in predefined_themes.items():
            self.add_theme(theme, prompt)

    def add_theme(self, theme_name: str, theme_prompt: str):
        self.themes[theme_name] = theme_prompt
        self._save_themes_to_file()

    def _save_themes_to_file(self):
        with open('themes.txt', 'w') as f:
            for theme, prompt in self.themes.items():
                f.write(f"{theme}: {prompt},\n")

    def load_themes_from_file(self):
        try:
            with open('themes.txt', 'r') as f:
                for line in f:
                    theme, prompt = line.strip().split(': ', 1)
                    self.themes[theme] = prompt
        except FileNotFoundError:
            print("Themes file not found. Using default themes.")

    def _fix_json_format(self, response):
        pattern = r'\[(.*?)\]'
        match = re.search(pattern, response, re.DOTALL)

        if match:
            extracted_content = match.group(0)

            return extracted_content
        else:
            print("No match found")

    def _format_llm_input(self, title: str, description: str, theme_prompt: str) -> str:
        formatted_prompt = f"{self.prompt_template}\nTitle: {title}\nDescription: {description}\nTheme: {theme_prompt}"
        print(formatted_prompt)
        return formatted_prompt

    def generate_meta_tags(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tagged_issues = []
        for issue in issues:
            tagged_issue = issue.copy()
            for theme in self.themes.keys():
                prompt = self._format_llm_input(
                    issue['Issue Title'],
                    issue['Description'],
                    f"{theme}: {self.themes.get(theme, 'General meta tags for the dialoguethat best describe the dialogue.')}"
                )
                response = self.llm_client.generate_response(prompt)
                # response = "['test']"
                fixed_response = self._fix_json_format(response)
                try:
                    tags = json.loads(fixed_response)
                    print(tags)
                    if isinstance(tags, list):
                        tagged_issue[f'{theme}_tags'] = ', '.join(tags)
                    else:
                        raise ValueError("Parsed JSON is not a list")
                except (json.JSONDecodeError, ValueError) as e:
                    print(
                        f"Error processing tags for issue: {issue['Issue Title']}, theme: {theme}")
                    print(f"Error: {str(e)}")
                    tagged_issue[f'{theme}_tags'] = ''
            tagged_issues.append(tagged_issue)
        return tagged_issues


def save_tagged_issues(tagged_issues: List[Dict[str, Any]], file_path: str):
    df = pd.DataFrame(tagged_issues)

    # Reorder columns
    columns = ['Issue Title', 'Description', 'Category']
    theme_columns = [col for col in df.columns if col.endswith('_tags')]
    columns.extend(theme_columns)

    df = df[columns]

    df.to_excel(file_path, index=False)
    print(f"Tagged issues saved to Excel file: {file_path}")


def load_issues(file_path: str) -> List[Dict[str, Any]]:
    df = pd.read_excel(file_path)
    return df.to_dict('records')


# Usage example
if __name__ == "__main__":
    # Choose your LLM client
    llm_client = GeminiClient()  # or OpenAIClient()

    # Create MetaTagGenerator
    tag_generator = MetaTagGenerator(llm_client)

    # Load themes from file (if exists)
    tag_generator.load_themes_from_file()

    # Add a custom theme (this will also save it to the file)
    tag_generator.add_theme(
        "workplace_innovation", "Identify innovative ideas and practices in the workplace.")

    # Load issues
    issues = load_issues('./test_dialogues.xlsx')

    # Initialize an empty DataFrame to store tagged issues
    df = pd.DataFrame(columns=['Issue Title', 'Description', 'Category'] +
                      [f'{theme}_tags' for theme in tag_generator.themes.keys()])

    # Generate meta tags for all themes and save immediately
    for issue in issues:
        tagged_issue = tag_generator.generate_meta_tags([issue])[0]

        # Convert the tagged_issue dictionary to a DataFrame and concatenate
        df = pd.concat([df, pd.DataFrame([tagged_issue])], ignore_index=True)

        # Save the updated DataFrame to Excel after each issue
        df.to_excel('./tagged_issues.xlsx', index=False)

        # Print results for the current issue
        print(f"Title: {tagged_issue['Issue Title']}")
        for theme in tag_generator.themes.keys():
            print(f"{theme} Tags: {tagged_issue[f'{theme}_tags']}")
        print("---")
        time.sleep(60)

    # print(f"Tagged issues saved to Excel file: ./tagged_issues.xlsx")
