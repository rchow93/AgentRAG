from phi.agent import Agent
from phi.knowledge.langchain import LangChainKnowledgeBase
from langchain_openai import OpenAIEmbeddings
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
import os
import logging
from dotenv import load_dotenv
from phi.model.openai import OpenAIChat
from phi.model.google import Gemini
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from langchain_chroma import Chroma

from confluence_tool import search_confluence_docs, retrieve_confluence_page
import traceback

####################
# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#####################
# Load environment variables
load_dotenv()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
SLACK_APP_LEVEL_TOKEN = os.getenv("SLACK_APP_LEVEL_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY2")
MODEL = "text-embedding-3-small"

# Chroma settings
CHROMA_PERSIST_DIR = "./chroma_db"
#####################

# Setup Slack app
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
bot_user_id = app.client.auth_test()["user_id"]


# Create knowledge retrievers for Chroma collections
def create_chroma_retriever(collection_name):
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model=MODEL)

    try:
        # Create Chroma retriever
        chroma_db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )

        # Set up the retriever with more results to compensate for no reranking
        retriever = chroma_db.as_retriever(search_kwargs={"k": 4})

        logger.info(f"Successfully created retriever for collection: {collection_name}")
        return retriever

    except Exception as e:
        logger.error(f"Error creating retriever for collection '{collection_name}': {str(e)}")
        return None


######setup retriever for first KB collection - Leadership #######
leadership_retriever = create_chroma_retriever("Leadership")
leadership_knowledge_base = LangChainKnowledgeBase(retriever=leadership_retriever)

######setup retriever for first KB collection - ArgoCD #######
argocd_retriever = create_chroma_retriever("ArgoCD")
argocd_knowledge_base = LangChainKnowledgeBase(retriever=argocd_retriever)

##########################################
leadership_knowledge_agent = Agent(
    name="Leadership Topics Knowledgebase RAG Agent",
    role="""You are an experienced knowledge finder with a true passionate for finding the answers to questions from users.
    As somebody with a data science background you are also very familiar with how vector databases are setup and how best to 
    retrieved answers from those data sources. You will always work to ensure that the best and most accurate answer is 
    found and wherever possible will include any citations or references included with those stored knowledge chunks. """,
    instructions="""Use the Leadership knowledge base to answer questions from the user about leadership. Use any tools or 
    methods that will retrieve the most accurate answer to the question generated from the Leadership Knowledgebase. Return 
    the output in a markdown format and structured in well written english. Treat this output professionally and with the utmost care.""",
    model=Gemini(id="gemini-2.0-flash-lite"),
    knowledge=leadership_knowledge_base,
    add_context=True,
    search_knowledge=True,
    markdown=True,
    debug_mode=True,
)

argocd_knowledge_agent = Agent(
    name="Argocd Topics Knowledgebase RAG Agent",
    role="""You are an experienced knowledge finder with a true passionate for finding the answers to questions from users.
    As somebody with a data science background you are also very familiar with how vector databases are setup and how best to 
    retrieved answers from those data sources. You will always work to ensure that the best and most accurate answer is 
    found and wherever possible will include any citations or references included with those stored knowledge chunks. """,
    instructions="""Use the Argocd knowledge base to answer questions from the user about Argocd. Use any tools or 
    methods that will retrieve the most accurate answer to the question generated from the Argocd Knowledgebase. Return 
    the output in a markdown format and structured in well written english. Treat this output professionally and with the utmost care.""",
    model=Gemini(id="gemini-2.0-flash-lite"),
    knowledge=argocd_knowledge_base,
    add_context=True,
    search_knowledge=True,
    markdown=True,
    debug_mode=True,
)

##########################################
web_search_agent = Agent(
    name="Web Search Agent",
    role=f"""You are a savvy technical researcher with smart skills to find any information on the Internet.
         You will be asked to use those skills to find the relevant information on a topic. You are passionate about that topic
         and will find the relevant information to share with the team. Use the available tools to find that information 
         to get the latest news and do not return old or stale data. Provide citations and links where possible. You're part of the 
         Mission Impossible Team with the responsibility to Search the web for the latest news and information.""",
    tools=[DuckDuckGo()],
    model=Gemini(id="gemini-2.0-flash-lite"),
)

finance_agent = Agent(
    name="Finance Agent",
    role=f""" You are a stock expert and investment stud. You have a knack for finding the best stocks to invest in.
    You will use your skills to find the relevant information on the latest stock prices, stock news, financials, and market trends.
    You are passionate about stocks and will find the relevant information to share with the team. Use the available tools to find that information.
    You are part of the Mission Impossible Team with the responsibility to find the latest stock prices and financial information. You are very
    familiar with stock/financials. You will use the available tools to find the latest stock prices and financial information. You will not provide
    old stale information because you know that could have negative consequences for the team. You will provide citations and links where possible.
    You will return: stock price, analyst recommendation, company info, stock fundamentals, income statements, key financial ratios, company news, 
    technical indicators, and historical prices. Your responsibility is to Handle financial queries, such as stock prices and market trends.""",
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, key_financial_ratios=True,
                         stock_fundamentals=True, income_statements=True, company_news=True, technical_indicators=True,
                         historical_prices=True, company_info=True)],
    model=Gemini(id="gemini-2.0-flash-lite"),
)

content_writer_agent = Agent(
    name="Expert Content Writer Agent",
    role=f"""Generates engaging blogs, articles, and other written content on topic.""",
    model=Gemini(id="gemini-2.0-flash-lite"),
)

confluence_agent = Agent(
    name="Confluence Knowledge Agent",
    role="""You are a specialized Confluence knowledge assistant. Your primary role is to help users find 
    relevant documentation and information stored in Confluence. You're equipped with tools to search 
    Confluence content and retrieve specific documents when needed.""",
    instructions=[
        "When users ask about documentation or information, use the search_confluence_docs tool to find relevant pages.",
        "Always include the URLs to the Confluence pages in your responses so users can access them directly.",
        "If the user asks for the full content, use the retrieve_confluence_page tool to get the full content of specific pages.",
        "Format your responses in a clear, organized way with markdown.",
        "If searching Confluence doesn't yield helpful results, acknowledge this and suggest alternatives or ask for more specific information.",
        "When you provide information from Confluence, cite the source by including the page title and URL."
    ],
    tools=[search_confluence_docs, retrieve_confluence_page],  # Using the new function name
    show_tool_calls=True,
    model=Gemini(id="gemini-2.0-flash-lite"),
    markdown=True,
    debug_mode=True,
)

##############################
router_agent = Agent(
    name="Router Agent",
    role="Routes user queries to the appropriate agent.",
    instructions=[
        "If the query is about Leadership, leadership, or related topics, route it to the Leadership Topics Knowledgebase RAG Agent.",
        "If the query is about ArgoCD, GitOps, or related topics, route it to the Argocd Topics Knowledgebase RAG Agent.",
        "If the query is about confluence documentation, confluence topics, or questions like 'where can I find information about X on confluence', route it to the Confluence Knowledge Agent.",

        "If the query is about recent news, weather, travel, or information, route it to the Web Search Agent.",
        "If the query is about financial stock, financial data, financial news, route it to the Finance Agent.",
        "If the query is about writing or generating content, route it to the Content Writer Agent.",

        "Absolutely - do not make anything up and do not provide old or stale information.",
    ],
    team=[leadership_knowledge_agent, argocd_knowledge_agent, confluence_agent, web_search_agent, finance_agent,
          content_writer_agent],
    show_tool_calls=True,
    add_history_to_messages=True,
    num_history_responses=8,
    model=OpenAIChat(id="gpt-4o"),
    markdown=True,
)


##################################################
@app.event("message")
def handle_message_events(event, say):
    user = event.get("user")
    text = event.get("text", "")
    channel = event.get("channel")
    thread_ts = event.get("thread_ts")  # Thread timestamp if message is in thread
    ts = event.get("ts")  # Message timestamp

    # Skip messages from the bot itself to avoid loops
    if user == bot_user_id:
        return

    # CASE 1: Direct mention in channel (starts a new thread)
    if not thread_ts and f"<@{bot_user_id}>" in text:
        # Create a new thread by using this message's ts
        process_and_respond(text, channel, ts, say, is_new_thread=True)
        return

    # CASE 2: Message in an existing thread
    if thread_ts:
        # Check if bot has participated in this thread before
        try:
            # Get conversation history for this thread
            result = app.client.conversations_replies(
                channel=channel,
                ts=thread_ts,
                limit=100  # Adjust based on your needs
            )

            # Check if bot has posted in this thread before
            bot_in_thread = any(message.get("user") == bot_user_id for message in result["messages"])

            if bot_in_thread:
                # Bot is part of this thread, respond without requiring mention
                process_and_respond(text, channel, thread_ts, say)
        except Exception as e:
            logger.error(f"Error checking thread participation: {str(e)}")
            logger.error(traceback.format_exc())


def process_and_respond(text, channel, thread_ts, say, is_new_thread=False):
    try:
        # Remove bot mention if present
        clean_input = text.replace(f"<@{bot_user_id}>", "").strip()

        if is_new_thread:
            # For new threads, acknowledge that we're processing
            say(
                text="Processing your request... I'll get back to you shortly.",
                channel=channel,
                thread_ts=thread_ts
            )

        logger.info(f"Processing request in {'new' if is_new_thread else 'existing'} thread: '{clean_input}'")

        # Get response from router agent
        agent_response = router_agent.run(clean_input)

        # Extract the string content from the RunResponse object
        if hasattr(agent_response, 'content'):
            response_text = agent_response.content
        else:
            response_text = str(agent_response)

        # Send the response back to Slack in the thread
        say(
            text=response_text,
            channel=channel,
            thread_ts=thread_ts
        )

    except Exception as e:
        # Log the error for debugging
        logger.error(f"Error processing message: {str(e)}")
        logger.error(traceback.format_exc())

        # Send a friendly error message to the user in the thread
        say(
            text=f"Sorry, I encountered an error while processing your request: {str(e)}",
            channel=channel,
            thread_ts=thread_ts
        )


if __name__ == "__main__":
    logger.info("Starting Socket Mode handler...")
    handler = SocketModeHandler(app, SLACK_APP_LEVEL_TOKEN)
    handler.start()


