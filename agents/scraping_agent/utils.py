# utils.py

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

visited = set()

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

def get_all_links(content, base_url):
    soup = BeautifulSoup(content, 'html.parser')
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        if is_valid(full_url):
            links.add(full_url)
    return links

def extract_financial_data(content):
    soup = BeautifulSoup(content, 'html.parser')
    financial_data = {}
    tables = soup.find_all('table')
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        rows = []
        for tr in table.find_all('tr'):
            cells = [td.get_text(strip=True) for td in tr.find_all('td')]
            if cells:
                rows.append(cells)
        if headers and rows:
            financial_data['headers'] = headers
            financial_data['rows'] = rows
            break  # Assuming the first relevant table is sufficient
    return financial_data

def crawl(url, max_depth=2, current_depth=0):
    if current_depth > max_depth or url in visited:
        return None
    visited.add(url)
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            content = response.text
            data = extract_financial_data(content)
            if data:
                return data
            links = get_all_links(content, url)
            for link in links:
                result = crawl(link, max_depth, current_depth + 1)
                if result:
                    return result
    except requests.RequestException as e:
        print(f"Request failed for {url}: {e}")
    return None
