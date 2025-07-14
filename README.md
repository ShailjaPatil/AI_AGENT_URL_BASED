## Author
**Shailja Patil**  
M.Tech CSE, IIT Kharagpur  
Passionate about Generative AI, LLMs, and building intelligent assistant tools
Feel free to connect or fork the repo to contribute or extend it further.

# URL-based Q&A Assistant (RAG Chatbot)
This project is a lightweight **RAG (Retrieval-Augmented Generation)** chatbot built using:
-  [Firecrawl](https://firecrawl.dev/) for website scraping
-  [LangChain](https://www.langchain.com/) for chaining components
-  [OpenRouter.ai](https://openrouter.ai) for hosted LLM inference (`deepseek/deepseek-chat-v3-0324:free`)
-  [Hugging Face](https://huggingface.co/inference-api) for generating **embeddings only** 
-  [ChromaDB](https://www.trychroma.com/) for vector storage
-  Flask for the backend API
-  HTML frontend interface
---

## Features
- Scrape any webpage via Firecrawl
- Convert scraped data into LLM-readable chunks and embeddings using Hugging Face embeddings
- Query with natural language using OpenRouter-hosted **DeepSeek Chat** model
- RAG pipeline with no local model downloads required
- Simple REST API with optional frontend
---

## LLM Model
This project uses the **`deepseek/deepseek-chat-v3-0324:free`** model hosted on [OpenRouter.ai](https://openrouter.ai):

## Requirements
- Python 3.10+
- `pip` or `venv`
- VS Code or any IDE
- Internet connection (for APIs)
---

## Setup
### 1. Clone the repo
bash
git clone (https://github.com/ShailjaPatil/AI_AGENT_URL_BASED.git)
cd AI_AGENT_URL_BASED

### 2. Create Virtual Enviornment
python -m venv venv
venv\Scripts\activate  

### 3. Install Dependencies
#1 Core framework and environment
flask
python-dotenv
#2 LangChain core
langchain
langchain-community
#3 Vector store and embeddings
chromadb
huggingface-hub
#4 OpenAI-compatible interface (used for OpenRouter)
openai
#5 Firecrawl for URL scraping
firecrawl

## Enviornment Variables in .env
#1 FIRECRAWL_API_KEY=your_firecrawl_key
#2 HF_TOKEN=your_huggingface_token

## USAGE
#1. Start flask APP
python app.py
http://127.0.0.1:5000/

#2 Open Browser
POST /scrape
Body: { "url": "(https://en.wikipedia.org/wiki/Prime_Minister_of_India)" }
POST /ask
Body: { "question": "Who is the current prime minister of india?" }

## Author
**Shailja Patil**  
M.Tech CSE, IIT Kharagpur  
Passionate about Generative AI, LLMs, and building intelligent assistant tools
Feel free to connect or fork the repo to contribute or extend it further.

## License
MIT License © 2025 Shailja Patil




