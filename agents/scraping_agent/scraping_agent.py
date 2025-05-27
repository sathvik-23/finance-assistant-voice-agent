import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.google import Gemini
from utils import crawl
from company_urls import company_urls

# Load environment variables from .env
load_dotenv()

# Access the API key
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the agent
agent = Agent(
    model=Gemini(
        id="gemini-2.0-flash",
        api_key=api_key,
        search=True
    ),
    show_tool_calls=True,
    markdown=True,
)

def summarize_company_financials(company_name):
    url = company_urls.get(company_name)
    if not url:
        print(f"URL for {company_name} not found.")
        return
    data = crawl(url)
    if data:
        prompt = f"Financial data extracted in english from {company_name}:\n\n{data}"
        agent.print_response(prompt)
        # agent.print_response(f"Financial data extracted from {company_name}:\n\n{data}")
    else:
        print(f"No financial data found for {company_name}.")

# Example usage
if __name__ == "__main__":
    company_name = "Tata Consultancy Services (TCS)"
    summarize_company_financials(company_name)
