import json
import re

import requests
from bs4 import BeautifulSoup


def fix_nested_json(raw_data):
    """
    Fixes potentially unbalanced or truncated nested JSON strings by ensuring all brackets are matched.

    Args:
        raw_data (str): Raw JSON string extracted from the HTML.

    Returns:
        str: Balanced JSON string.
    """
    stack = []
    for i, char in enumerate(raw_data):
        if char == '[':
            stack.append('[')
        elif char == ']':
            if stack:
                stack.pop()
        # If stack is empty and we've reached a closing bracket, assume end of JSON
        if not stack and char == ']':
            return raw_data[:i + 1]
    return raw_data  # Return as is if no issues detected


class HFCrawler:

    def __init__(self, url, start_page, end_page):
        self.models_url = url
        self.start_page = start_page
        self.end_page = end_page

    def extract_complete_models_list(sefl, page_url):
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(page_url, headers=headers)
        if response.status_code != 200:
            print(f"Error: Unable to fetch the page (status code {response.status_code}).")
            return []

        soup = BeautifulSoup(response.content, "html.parser")

        # Search for 'models' key in the entire HTML
        match = re.search(r'"models":(\[.*\])', str(soup), re.DOTALL)
        if match:
            raw_data = match.group(1)  # Extract the JSON array
            try:
                # Fix unbalanced brackets by using a JSON decoder
                balanced_data = fix_nested_json(raw_data)
                models_list = json.loads(balanced_data)  # Convert JSON string to Python list
                return models_list
            except Exception as e:
                print(f"Error decoding JSON: {e}")
        else:
            # match = re.search(r'"text-smd">(\[.*\])</h4>', str(soup), re.DOTALL)
            # print(match)
            repo_name_tags = soup.find_all('h4',
                                           class_='text-md truncate font-mono text-black dark:group-hover/repo:text-yellow-500 group-hover/repo:text-indigo-600 text-smd')
            repo_names = [tag.get_text(strip=True) for tag in repo_name_tags]
            return [{"id": _id} for _id in repo_names]

        print("No 'models' list found.")
        return []

    def get_model_info(sefl):
        model_metadata = {}
        model_ids = []
        page_urls = []
        if sefl.start_page is None:
            page_urls = [sefl.models_url]
        else:
            for i in range(sefl.start_page, sefl.end_page):
                print(i)
                page_urls.append(f"{sefl.models_url}&p={i}")

        for page_url in page_urls:
            models_metadata = sefl.extract_complete_models_list(page_url)
            for model_data in models_metadata:
                model_id = model_data.get("id")
                model_ids.append(model_id)
                model_metadata[model_id] = model_data

        return model_ids, model_metadata


if __name__ == '__main__':
    crawler = HFCrawler(
        "https://huggingface.co/models?other=base_model:finetune:google%2Fvit-large-patch16-224-in21k&sort=downloads", 0,
        2)
    model_ids, model_metadata = crawler.get_model_info()
    print(model_ids)
