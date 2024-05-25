

from langchain.prompts import ChatPromptTemplate , HumanMessagePromptTemplate,MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


load_dotenv()

chat  = ChatOpenAI(verbose=True)

memory = ConversationBufferMemory(memory_key="messages", return_messages=True)

chatPromotTemplate = ChatPromptTemplate(
      input_variables=['content','messages'],
      messages=[
          MessagesPlaceholder(variable_name="messages"),
          HumanMessagePromptTemplate.from_template('{content}')
      ]
)

chain = LLMChain(
    llm = chat,
    prompt = chatPromotTemplate,
    verbose = True,
    memory = memory
)


while True:
    content = input(">>")
    result  = chain({
        'content' : content
    })
    print(result['text'])
    
    