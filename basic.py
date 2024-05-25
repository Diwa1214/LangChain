from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


llm = OpenAI()

result = llm("Do you know about gilli movie ?")

print(result)


