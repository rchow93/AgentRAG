Here's a `README.md` file for your project, explaining the functionality and usage of both `slacker.py` (Slack-based agentic agent) and `dataloader.py` (Chroma-based data loader for PDFs and EPUBs).  

```markdown
# Slack-Based Agentic AI with Knowledgebase and Q&A  

This project integrates an **Agentic AI system** into Slack, enabling knowledge retrieval, Q&A, and tool-based interactions. It leverages **retrieval-augmented generation (RAG)** with **ChromaDB**, **OpenAI embeddings**, and multiple AI agents.  

## Features  
- **Slack Integration**: Users can ask questions directly in Slack, and the bot responds intelligently.  
- **Agentic AI**: Different agents handle topics like leadership, ArgoCD, finance, content writing, web search, and Confluence documentation.  
- **Knowledge Retrieval**: Uses **ChromaDB** as a vector database to fetch relevant information.  
- **RAG (Retrieval-Augmented Generation)**: Ensures responses are backed by indexed knowledge sources.  
- **Multi-Model Support**: Uses **Gemini (Google AI)** and **GPT-4o-mini** for intelligent responses.  
- **Confluence Search**: Retrieves company documentation for enterprise users.  

---

## 1️⃣ Slack Bot (`slacker.py`)  

### Overview  
The **Slack-based Agentic AI** listens to messages in Slack channels and intelligently routes them to specialized AI agents. It uses:  
- `Slack Bolt` for message handling  
- `Phi.Agent` for managing AI agents  
- `ChromaDB` for vector-based knowledge retrieval  
- `OpenAIEmbeddings` for efficient search  
- `Gemini` and `GPT-4o-mini` models for intelligent responses  

### Setup Instructions  
To Create a SlackBot:
slack: https://www.pragnakalp.com/create-slack-bot-using-python-tutorial-with-examples/
slack: https://medium.com/@davidjohnakim/create-an-agebot-in-slack-using-python-fb6aa41e1826

#### **Step 1: Install Dependencies**  
```sh
pip install slack_bolt langchain_openai langchain_chroma phi dotenv
```

#### **Step 2: Set Up Environment Variables**  
Create a `.env` file and configure:  
```ini
SLACK_BOT_TOKEN=your_slack_bot_token
SLACK_SIGNING_SECRET=your_slack_signing_secret
SLACK_APP_LEVEL_TOKEN=your_slack_app_level_token
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

#### **Step 3: Run the Slack Bot**  
```sh
python slacker.py
```
The bot will listen for Slack messages and respond with intelligent answers.  

### **Agentic AI Workflow**  
- **Router Agent**: Routes queries to the correct AI agent.  
- **Leadership & ArgoCD Agents**: Answer domain-specific questions using ChromaDB.  
- **Web Search Agent**: Uses DuckDuckGo for real-time web search.  
- **Finance Agent**: Fetches stock data via `yfinance`.  
- **Confluence Agent**: Searches Confluence documentation for company knowledge.  
- **Content Writer Agent**: Assists in writing articles and summaries.  

---

## 2️⃣ Data Loader (`dataloader.py`)  

### Overview  
This script **indexes knowledge from EPUB and PDF files** into **ChromaDB**, enabling the Slack bot to retrieve accurate responses via RAG.  

### Setup Instructions  

#### **Step 1: Install Dependencies**  
```sh
pip install langchain_openai langchain_chroma langchain_community dotenv
```

#### **Step 2: Prepare Your Documents**  
Place your **EPUB** and **PDF** files inside `./docs`, organized in subdirectories based on topic:  
```
docs/
├── Leadership/
│   ├── leadership_book.pdf
│   └── management.epub
├── ArgoCD/
│   ├── argocd_guide.pdf
│   └── deployment.epub
```

#### **Step 3: Run the Data Loader**  
```sh
python dataloader.py
```
This script will:  
- Extract text from **EPUB/PDF** files  
- Split the content into manageable chunks  
- Store embeddings in **ChromaDB** for later retrieval  

### **ChromaDB Storage**  
The data is saved in:  
```
./chroma_db/
```
Each subdirectory in `docs/` becomes a **separate collection** in ChromaDB.

---

## 3️⃣ How It Works  

1. **Load Documents** → `dataloader.py` processes EPUB/PDF files into ChromaDB.  
2. **Query in Slack** → Users ask questions, and `slacker.py` routes them to the right agent.  
3. **Retrieve Knowledge** → The agent fetches relevant information from ChromaDB or online sources.  
4. **Generate Response** → AI generates an answer and replies in Slack.  

---

## 4️⃣ Future Improvements  
- ✅ Add support for DOCX and Markdown files  
- ✅ Improve response ranking using **re-ranking models**  
- ✅ Expand Confluence integration for better enterprise search  

---

## 5️⃣ Credits  
Built with ❤️ using **Slack Bolt, LangChain, ChromaDB, OpenAI, and Google Gemini**. 🚀
```
📄 License
This project is licensed under the Apache License - see the LICENSE file for details.

This README provides a **clear walkthrough** of both scripts, their purpose, setup, and how the system works. Let me know if you'd like any refinements! 🚀