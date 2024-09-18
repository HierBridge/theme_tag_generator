import random
import re
import json
import time
from typing import List, Dict, Any


class MetaTagGenerator:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.prompt_template = self._load_prompt_template()
        self.themes = self._load_themes()

    def _load_prompt_template(self) -> str:
        with open('./generic_prompt.txt', 'r') as f:
            return f.read()

    def _load_themes(self) -> Dict[str, str]:
        """
        Load themes from a file or define them here.
        Each theme has a name and a description or prompt.
        """
        # For simplicity, define themes here. Alternatively, load from a file.
        themes = {
            "employee_individual": "Find or suggest closely related words for individual employee performance, skills, and personal development.",
            "team_performance": "Find or suggest closely related words for team dynamics, collaboration, and collective achievements.",
            "organizational_commitment": "Find or suggest closely related words for employee loyalty, dedication, and alignment with company values.",
            "organizational_citizenship_behavior": "Find or suggest closely related words for extra-role behaviors, volunteerism, and actions that benefit the organization beyond job requirements."
        }
        return themes

    def format_prompt(self, issue_title: str, description: str, theme_name: str) -> str:
        theme_prompt = self.themes.get(theme_name, '')
        formatted_prompt = self.prompt_template.replace('{Issue Title}', issue_title)\
                                               .replace('{Description}', description)\
                                               .replace('{Theme}', theme_prompt)
        return formatted_prompt

    def extract_tags(self, response: str) -> List[str]:
        # Extract JSON array from response
        pattern = r'\[.*?\]'
        match = re.search(pattern, response, re.DOTALL)
        if match:
            extracted_content = match.group(0)
            try:
                tags = json.loads(extracted_content)
                if isinstance(tags, list):
                    return [tag.strip() for tag in tags]
                else:
                    return []
            except json.JSONDecodeError:
                return []
        else:
            return []

    def generate_meta_tags(self, issue_title: str, description: str) -> Dict[str, List[str]]:
        """
        Generate meta tags for each theme.
        Returns a dictionary with themes as keys and list of tags as values.
        """
        tags_per_theme = {}
        for theme_name in self.themes.keys():
            prompt = self.format_prompt(issue_title, description, theme_name)
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.llm_client.generate_response(prompt)
                    tags = self.extract_tags(response)
                    tags_per_theme[theme_name] = tags
                    print(f"Generated tags for theme '{theme_name}'")
                    break  # Break out of retry loop on success
                except Exception as e:
                    if attempt == max_retries:
                        print(
                            f"Failed after {max_retries} attempts for theme '{theme_name}': {e}")
                        tags_per_theme[theme_name] = []
                    else:
                        wait_time = 2 ** attempt + random.uniform(0, 1)
                        print(
                            f"Attempt {attempt} failed for theme '{theme_name}'. Retrying in {wait_time:.2f} seconds...")
                        time.sleep(wait_time)
            # Sleep to respect rate limits
            time.sleep(15)  # Adjust sleep time as needed
        return tags_per_theme
