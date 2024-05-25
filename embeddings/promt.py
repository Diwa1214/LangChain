from langchain_chroma import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA,LLMChain
from langchain.globals import set_debug
from dotenv import load_dotenv

set_debug(True)

load_dotenv()

# Loading Chain

llm = OpenAI()

# Creating Embedding
embedding = OpenAIEmbeddings()

# Getting Chromba DB instance 
db = Chroma(
    embedding_function= embedding,
    persist_directory="embeddings/db"
)

# gettting Retriever
retriever = db.as_retriever()


chains = RetrievalQA.from_chain_type(
    llm = llm,
    retriever = retriever,
    chain_type = "stuff"
)

result = chains.run("What is most interesting fact on English Language ?")

print(result)
