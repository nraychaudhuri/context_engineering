from exa_py import Exa
from typing import Any
import os
from dotenv import load_dotenv

load_dotenv()


def exa_search(query: str) -> list[dict[str, str]]:
    exa = Exa(api_key=os.getenv("EXA_API_KEY"))
    result = exa.search_and_contents(query, text=True)
    news = result.results
    response = []
    for n in news:
        response.append(
            {
                "url": n.url,
                "title": n.title,
                "summary": n.text[:100],
            }
        )
    # print(">>>>")
    # print(response)
    # print(">>>>")
    return response
