
from langchain.prompts import ChatPromptTemplate , HumanMessagePromptTemplate,MessagesPlaceholder
from  langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI(verbose=True)


memory = ConversationSummaryBufferMemory(
    memory_key="messages",
    return_messages=True,
    llm= chat
)

chatPromtTemplate = ChatPromptTemplate(
    input_variables=['content','messages'],
    messages= [
        MessagesPlaceholder(variable_name='messages'),
        HumanMessagePromptTemplate.from_template('{content}')
    ]
)

chain  = LLMChain(
    prompt = chatPromtTemplate,
    llm = chat,
    verbose = True,
    memory = memory
)

while True:
    content = input(">>")
    result = chain({
        'content':content
    })
    
    print(result['text'])