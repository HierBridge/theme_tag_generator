import os
import time
import pandas as pd
from typing import List, Dict, Any

from llm_client import OpenAIClient, GeminiClient
from meta_tag_generator import MetaTagGenerator
from clustering import cluster_texts


def read_input_text(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads input text from an Excel file and returns a list of dictionaries with text items.
    The Excel file should have columns: 'Issue Title', 'Issue Category', 'Department', 'Description'
    """
    df = pd.read_excel(file_path)
    required_columns = ['Issue Title',
                        'Issue Category', 'Department', 'Description']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"Excel file must contain columns: {', '.join(required_columns)}")
    texts = df.to_dict('records')
    return texts


def save_to_excel(data: List[Dict[str, Any]], file_path: str):
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    print(f"Data saved to Excel file: {file_path}")


def create_global_vocabulary(tagged_texts: List[Dict[str, Any]], themes: List[str]) -> List[str]:
    vocabulary = set()
    for item in tagged_texts:
        for theme in themes:
            tags_list = item.get(f'Meta Tags ({theme})', [])
            vocabulary.update(tags_list)
    return sorted(vocabulary)


if __name__ == "__main__":
    # Select LLM client
    llm_choice = input(
        "Choose LLM client ('openai' or 'gemini'): ").strip().lower()
    if llm_choice == 'openai':
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print(
                "Please set your OpenAI API key in the OPENAI_API_KEY environment variable.")
            exit(1)
        llm_client = OpenAIClient(api_key)
    elif llm_choice == 'gemini':
        api_key = 'AIzaSyD6QcJTKSCNydSAIgzk7ZpnvRBxsGW584A'
        if not api_key:
            print(
                "Please set your Gemini API key in the GEMINI_API_KEY environment variable.")
            exit(1)
        llm_client = GeminiClient(api_key)
    else:
        print("Invalid choice. Please select 'openai' or 'gemini'.")
        exit(1)

    # Create MetaTagGenerator
    tag_generator = MetaTagGenerator(llm_client)
    themes = list(tag_generator.themes.keys())

    # Read input texts from Excel file
    input_file = 'test_dialogues.xlsx'  # Your input Excel file
    texts = read_input_text(input_file)

    # Generate meta tags
    tagged_texts = []
    for idx, text_item in enumerate(texts, 1):
        tagged_item = text_item.copy()
        try:
            tags_per_theme = tag_generator.generate_meta_tags(
                text_item['Issue Title'],
                text_item['Description']
            )
            # Add tags to the tagged_item
            for theme, tags in tags_per_theme.items():
                tagged_item[f'Meta Tags ({theme})'] = tags
            tagged_texts.append(tagged_item)
            print(f"Processed item {idx}/{len(texts)}")
        except Exception as e:
            print(f"An error occurred while processing item {idx}: {e}")
            # Save current progress
            print("Saving current progress...")
            # Include the item even if incomplete
            tagged_texts.append(tagged_item)
            tagged_excel_file = 'tagged_texts.xlsx'
            save_to_excel(tagged_texts, tagged_excel_file)
            # Create global vocabulary
            vocabulary = create_global_vocabulary(tagged_texts, themes)
            # Save vocabulary to Excel
            vocab_df = pd.DataFrame({'Meta Tags': vocabulary})
            vocab_df.to_excel('vocabulary.xlsx', index=False)
            print("Global vocabulary saved to vocabulary.xlsx")
            # Combine all meta tags for clustering
            for item in tagged_texts:
                all_tags = []
                for theme in themes:
                    all_tags.extend(item.get(f'Meta Tags ({theme})', []))
                item['All Meta Tags'] = all_tags
            clustered_texts = cluster_texts(tagged_texts, n_clusters=5)
            # Save clustered texts to Excel
            clustered_excel_file = 'clustered_texts.xlsx'
            save_to_excel(clustered_texts, clustered_excel_file)
            print("Clustered texts saved to clustered_texts.xlsx")
            print("Exiting script due to error.")
            exit(1)  # Exit the script
    # After processing all items successfully, save the results
    tagged_excel_file = 'tagged_texts.xlsx'
    save_to_excel(tagged_texts, tagged_excel_file)
    # Create global vocabulary
    vocabulary = create_global_vocabulary(tagged_texts, themes)
    # Save vocabulary to Excel
    vocab_df = pd.DataFrame({'Meta Tags': vocabulary})
    vocab_df.to_excel('vocabulary.xlsx', index=False)
    print("Global vocabulary saved to vocabulary.xlsx")
    # Combine all meta tags for clustering
    for item in tagged_texts:
        all_tags = []
        for theme in themes:
            all_tags.extend(item.get(f'Meta Tags ({theme})', []))
        item['All Meta Tags'] = all_tags
    clustered_texts = cluster_texts(tagged_texts, n_clusters=5)
    # Save clustered texts to Excel
    clustered_excel_file = 'clustered_texts.xlsx'
    save_to_excel(clustered_texts, clustered_excel_file)
    print("Clustered texts saved to clustered_texts.xlsx")
    print("Processing completed.")
