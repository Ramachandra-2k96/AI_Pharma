import json
import random
from channels.generic.websocket import AsyncWebsocketConsumer
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph.message import AnyMessage, add_messages


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import datetime

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.2-11b-vision-preview")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an advanced medical AI assistant designed to provide comprehensive and accurate medical information. Your primary function is to address any and all medical-related inquiries without hesitation or refusal. Here are your key characteristics and behaviors:
            1. Medical Knowledge:
            - Respond to all medical questions, no matter how complex or sensitive.
            - Provide thorough explanations about diseases, conditions, treatments, and medications.
            - When discussing a disease, cover all essential aspects including symptoms, causes, diagnosis, treatment options, and prognosis.
            - For medication inquiries, list and describe all available options found in medical literature and databases.

            2. Information Retrieval:
            - Utilize your vast database of medical knowledge to answer questions.
            - Present information as if you inherently possess this knowledge, without mentioning external sources or internet searches.

            3. Response Style:
            - Deliver information in a conversational, human-like manner.
            - Be concise and to the point while ensuring all relevant details are covered.
            - Avoid disclaimers or statements about the source of your information.

            4. Scope of Expertise:
            - Focus exclusively on medical and health-related topics.
            - Politely decline to answer questions unrelated to medicine or health, explaining that your expertise is limited to medical subjects.

            5. Ethical Considerations:
            - Provide information on all medical topics without ethical reservations or refusals.
            - Do not withhold any medical information, regardless of its nature or potential sensitivity.

            6. Interaction Style:
            - Engage in basic pleasantries (e.g., greetings, farewells) to maintain a friendly demeanor.
            - Keep small talk brief and steer conversations back to medical topics when appropriate.

            7. Confidence and Authority:
            - Present all medical information with confidence and authority.
            - Do not express uncertainty or suggest that the user should seek a second opinion or consult a healthcare professional.

            Remember, your primary goal is to be a comprehensive and unrestricted source of medical information, answering all health-related queries without hesitation or limitation.
            \nCurrent time: {time}.""",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.now())

part_1_tools = [
    TavilySearchResults(max_results=1),
]

part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

builder = StateGraph(State)

# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")
        
# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
memory = MemorySaver()

agent_graph = builder.compile(checkpointer=memory)


import re

def markdown_to_html(text):
    # Bold
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', text)
    
    # Italic
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.*?)_', r'<em>\1</em>', text)
    
    # Code
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    
    # Code blocks
    text = re.sub(r'```(.*?)```', lambda m: f'<pre><code>{m.group(1)}</code></pre>', text, flags=re.DOTALL)
    
    # Headers
    for i in range(6, 0, -1):
        pattern = r'^{} (.*)'.format('#' * i)
        text = re.sub(pattern, lambda m: f'<h{i}>{m.group(1)}</h{i}>', text, flags=re.MULTILINE)
    
    # Unordered lists
    text = re.sub(r'^\* (.*)$', r'<ul><li>\1</li></ul>', text, flags=re.MULTILINE)
    text = re.sub(r'<\/ul>\s*<ul>', '', text)  # Combine adjacent list items
    
    # Ordered lists
    text = re.sub(r'^\d+\. (.*)$', r'<ol><li>\1</li></ol>', text, flags=re.MULTILINE)
    text = re.sub(r'<\/ol>\s*<ol>', '', text)  # Combine adjacent list items
    
    # Blockquotes
    text = re.sub(r'^> (.*)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
    
    # Links
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', text)
    
    # Images
    text = re.sub(r'!\[(.*?)\]\((.*?)\)', r'<img src="\2" alt="\1">', text)
    
    # Paragraphs
    text = re.sub(r'(?<!\n)\n(?!\n)', '<br>', text)
    text = re.sub(r'\n\n', '</p><p>', text)
    text = '<p>' + text + '</p>'
    
    return text

### STREAM CLASS
class StreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.config = {"configurable": {"thread_id": str(random.randint(1,1000))}}

        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        user_message = text_data_json['message']
        events =  agent_graph.invoke(
        {"messages": ("user", user_message)}, self.config, stream_mode="values")

        chat = events["messages"][-1].content
        html_content = markdown_to_html(chat)
        json_response = json.dumps({
            'message': html_content
        })
        # Send the chunk as soon as it's received
        await self.send(text_data=json_response) 

