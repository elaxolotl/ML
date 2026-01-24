import requests
from bs4 import BeautifulSoup
import pandas as pd

BASE_URL = "https://scancode-licensedb.aboutcode.org/"

def scrape_json_links(url):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    response = session.get(url, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    rows = []
    links = soup.select("a")

    json_links = [
        BASE_URL + a["href"]
        for a in links
        if a.text.strip() == "json"
    ]

    print("Found:", len(json_links), "json links")

    for link in json_links:
        try:
            r = session.get(link, timeout=10)
            data = r.json()
            rows.append({
                "key": data.get("key"),
                "text": data.get("text"),
                "category": data.get("category")
            })
        except requests.exceptions.JSONDecodeError:
            print(f"skipped invalid json from {link}")
            continue

    return pd.DataFrame(rows)

df = scrape_json_links(BASE_URL)
df.to_csv('data.csv', index=False)