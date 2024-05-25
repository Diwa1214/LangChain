# Building the chat model 

# 1 . Create the chatPromotTemplate 
# 2 . pass through the model

from langchain.prompts import ChatPromptTemplate , HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

chatPromotTemplate = ChatPromptTemplate(
       input_variables=['content'],
        messages= [
            HumanMessagePromptTemplate.from_template("{content}")
        ]
)

chat  = ChatOpenAI()


chain = LLMChain(
    llm = chat,
    prompt = chatPromotTemplate,
)




while True:
    content = input(">>")
    
    result = chain({
        'content':content
    })
    
    print(result['text'])
    