from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain , SequentialChain
import argparse
from dotenv import load_dotenv

# Load the env variables
load_dotenv()


parser = argparse.ArgumentParser()

parser.add_argument('--language',default="c++")

parser.add_argument('--task', default="return hello world")

args = parser.parse_args()

# LLM Define
llm = OpenAI()

# Create a prompt Template
code_prompt = PromptTemplate(
    template= "Write a short {language} function  that will {task}",
    input_variables=['language','task'],
)


# Create a second Prompt Template

second_code_prompt = PromptTemplate(
    template= "Based follwing {language} and code {code} give the dummy code exmaple for this",
    input_variables=['language','code'],
)

# Create a First  chain 
code_chain = LLMChain(
    llm = llm,
    prompt = code_prompt,
    output_key = 'code'
)

# Create a Second Chain
second_code_chain = LLMChain(
    llm = llm,
    prompt = second_code_prompt,
    output_key = 'code2'
)

# Combine Two chain together

chains = SequentialChain(
    chains = [code_chain,second_code_chain],
    input_variables = ['language','task'],
    output_variables = ['code','code2']
)

result = chains({
    'language':args.language,
    'task' : args.task
})

print('>>>> GENERATED CODE')

print(result['code'])


print(">>>> GENERATED EXAMPLE")

print(result['code2'])