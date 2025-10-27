from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.memory import ConversationBufferMemory
import http.client
from openai import OpenAI
import json
import os
import requests
from langchain.tools import BaseTool
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import quote
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()
#adding the wikipedia searching tool
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

#defining the function that returns the data of the employees
@tool
def get_employee_details(employee_id: str) -> dict:
    """Simulated function to get employee details."""
    employee_db = {
        "E001": {"name": "Jihad Barakat", "position": "Stocker", "department": "Chocolate and chips"},
        "E002": {"name": "Ali Sharif", "position": "Delivery picker/Cashier", "department": "Online orders"},
        "E003": {"name": "Ihab", "position": "Manager", "department": "Store Management"},
        "E004": {"name": "David Kim", "position": "Product Manager", "department": "Product"},
        "E005": {"name": "Emma Davis", "position": "UX Designer", "department": "Design"},
        "E006": {"name": "Frank Moore", "position": "DevOps Engineer", "department": "IT"},
        "E007": {"name": "Grace Liu", "position": "QA Engineer", "department": "QA"},
        "E008": {"name": "Henry Walker", "position": "Sales Representative", "department": "Sales"},
        "E009": {"name": "Irene Chen", "position": "Marketing Specialist", "department": "Marketing"},
        "E010": {"name": "Jack Brown", "position": "Business Analyst", "department": "Finance"},
        "E011": {"name": "Karen Green", "position": "Recruiter", "department": "HR"},
        "E012": {"name": "Liam White", "position": "Intern", "department": "IT"},
        "E013": {"name": "Maya Patel", "position": "Solutions Architect", "department": "IT"},
        "E014": {"name": "Noah Rivera", "position": "Legal Counsel", "department": "Legal"},
        "E015": {"name": "Olivia Wilson", "position": "Chief Financial Officer", "department": "Finance"},
    }
    return employee_db.get(employee_id, {"error": "Employee not found"})

#defining the function that returns the eave days of the eamployee
@tool
def check_leave_balance(employee_id: str) -> dict:
    """Simulated function to check leave balance for an employee."""
    leave_db = {
        "E001": {"annual_leave": 10, "sick_leave": 5},
        "E002": {"annual_leave": 15, "sick_leave": 7},
        "E003": {"annual_leave": 12, "sick_leave": 6},
        "E004": {"annual_leave": 14, "sick_leave": 4},
        "E005": {"annual_leave": 11, "sick_leave": 8},
        "E006": {"annual_leave": 13, "sick_leave": 5},
        "E007": {"annual_leave": 9,  "sick_leave": 6},
        "E008": {"annual_leave": 16, "sick_leave": 7},
        "E009": {"annual_leave": 10, "sick_leave": 5},
        "E010": {"annual_leave": 12, "sick_leave": 4},
        "E011": {"annual_leave": 15, "sick_leave": 10},
        "E012": {"annual_leave": 5,  "sick_leave": 2},
        "E013": {"annual_leave": 14, "sick_leave": 6},
        "E014": {"annual_leave": 13, "sick_leave": 5},
        "E015": {"annual_leave": 18, "sick_leave": 8},
    }

    return leave_db.get(employee_id, {"error": "Employee not found"})


    



#setting the tools and prompt for the agent
tools=[get_employee_details, check_leave_balance,wikipedia_tool]

prompt = ChatPromptTemplate.from_messages(
    [
        ("system" , "You are a HR assistant. Use tools when needed.only answer questions related to HR and tech industry."),
        ("system" , "use the tool named wikipedia_tool when you want to search for and generate intervoew questions for the job role or to get any external informations to answer the user."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder" , "{agent_scratchpad}")
    ]
)
#setting up the llm that will power the agent
llm = ChatOpenAI(model="gpt-4-turbo", api_key=api_key)
#setting up the memory for the agent
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#creating the agent
agent = create_tool_calling_agent(llm, tools, prompt)
#creating the agent executor
agent_executor = AgentExecutor(
    agent = agent,
    tools = tools,
    memory = memory,
    verbose=True
)
#function to interact with the agent
def HR_assistant(message, history):
    response = agent_executor.invoke({"input" :message})
    return response['output']
#gradio interface to chat with the agent
if __name__ == "__main__":
    import gradio as gr
    iface = gr.ChatInterface(
        fn = HR_assistant,
        title="💬 HR Assistant Chat",
        description="Chat with an AI!",
    )
    iface.launch()
