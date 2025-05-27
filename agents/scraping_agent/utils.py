# utils.py

import requests
from bs4 import BeautifulSoup

def fetch_website_content(url):
    """
    Fetches and returns the main text content from the specified URL.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None
