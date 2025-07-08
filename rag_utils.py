# UTIL CLASS CONTAINING UTILITY FUCNTIONS 

from dotenv import load_dotenv
load_dotenv()
from firecrawl import AsyncFirecrawlApp
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# FUNCTION TO SCRAPE THE URL
async def scrape_url(url):
    app = AsyncFirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))
    response = await app.scrape_url(
        url=url,
        formats=['markdown'],  
        only_main_content=True
    )
    return response.markdown  

# FUCNTION TO SPLIT THE TEXT INTO DOCUMENTS CHUNKS
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.create_documents([text])

# FUNCTION TO EMBED THE DOCS 
def embed_documents(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  
        model_kwargs={'device': 'cpu'},  
        encode_kwargs={'normalize_embeddings': True}
    )
    db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory="./chroma_db" 
    )
    return db