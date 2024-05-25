
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
import langchain_core


load_dotenv()

# Load the document
loader = TextLoader('embeddings/facts.txt')

# Creating chunk object
text_spliiter = CharacterTextSplitter(
    separator='\n',
    chunk_size = 200,
    chunk_overlap = 0
)

# Load and  Creating a chunks
chunks = loader.load_and_split(
    text_splitter= text_spliiter
)

# creating embedding
embeddings = OpenAIEmbeddings()


# creating vector db with using embedding

db = Chroma.from_documents(
    documents= chunks,
    embedding= embeddings,
    persist_directory='embeddings/db',
)

results  = db.similarity_search_with_score("What is most interesting fact on English Language ?")

print(results)

print(">>> Generate >>> ")


for result in results:
    print('\n')
    print(result[1])
    print(result[0].page_content)




