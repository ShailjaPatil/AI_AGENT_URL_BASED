# MAIN APP FILE TO HANDLE API FUNCTIONS  AND  LLM

# FOR LANGCHAIN AND LLM
from dotenv import load_dotenv
load_dotenv()
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
import os 

# FOR API HANDELING
from flask import Flask, request, jsonify, send_from_directory
from rag_utils import scrape_url, split_text, embed_documents 
import asyncio

from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

import shutil
from pathlib import Path

def clear_chroma_cache():
    chroma_path = Path("./chroma_db")  # Your persist directory
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
        print("✅ ChromaDB cache deleted")
    else:
        print("⚠️ No ChromaDB cache found")

# Call this before creating new embeddings
clear_chroma_cache()

# Your OpenRouter API key (get free key: https://openrouter.ai/keys)
sec_key = os.getenv("SEC_KEY")

llm = ChatOpenAI(
    model_name="deepseek/deepseek-chat-v3-0324:free",  # Free model
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=sec_key,  # Directly using sec_key
    max_tokens=256,
    temperature=0.7
)

# Test LLM connection
try:
    test_response = llm.invoke([HumanMessage(content="Hello")])
    print("LLM Connection Test:", test_response.content)
except Exception as e:
    print("LLM Connection Failed:", str(e))


# LLM USING HUGGINGFACE ENDPOINT
# llm = HuggingFaceEndpoint(
#     repo_id="HuggingFaceH4/zephyr-7b-beta",
#     task="text-generation",
#     huggingfacehub_api_token=os.getenv("HF_TOKEN"),
#     max_new_tokens=256,
#     # temperature=0.3,
#     # repetition_penalty=1.2,  
# )

# PROMPT GENERATION  TEMPLATE
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    <|system|>
    You are a context-only AI assistant. Follow these rules STRICTLY:
    1. Answer ONLY using the information from the provided context
"
</s>

    Context: {context}
    Question: {question}
    Answer:
    """
)

# LANGCHAIN RetrievalQA
def get_qa_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("\n inside get_qa_chain")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

# APP STARTING POINT 
app = Flask(__name__)
qa_chain = None

# ROUTE TO SCRAPE URL AND MAKE EMBEDDINGS
@app.route("/scrape", methods=["POST"])
def scrape():
    clear_chroma_cache() 
    print("\nCACHE CLEARED")
    url = request.json["url"]
    print("\nurl = "+ url)
    text = asyncio.run(scrape_url(url))
    print("\n SCRAPING DONE  SUCCESSFULLY")
    docs = split_text(text)
    print("\n DOCS CREATED SUCCESSFULLY")
    global qa_chain
    qa_chain = get_qa_chain(embed_documents(docs))
    print("\nEMBEDDING DONE AND  CHAIN CREATED SUCCESSFULLY")
    return jsonify({"message": "Cache cleared ,Scraping and embedding done."})
    
# ROUTE TO ASK QUESTION AND RESPOND
@app.route("/ask", methods=["POST"])
def ask():
    question = request.json["question"]
    print("\n QUESTION : "+ question)
    if qa_chain is None:
        return jsonify({"error": "No data embedded yet."}), 400
    result = qa_chain.invoke({"query": question})
    
    answer_text = result["result"] if isinstance(result, dict) else result
    print("\n ANSWER : "+ answer_text)
    return jsonify({"answer": answer_text})

# ROUTE TO THE HOMEPAGE
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

if __name__ == "__main__":
    app.run(debug=True)