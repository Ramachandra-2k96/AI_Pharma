from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import AnyMessage, add_messages
import os
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["TAVILY_API_KEY"] = tavily_api_key

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from datetime import datetime

from langchain_groq import ChatGroq
llm = ChatGroq(model="llama-3.2-11b-vision-preview")


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

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda

from langgraph.prebuilt import ToolNode

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

def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition

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
part_1_graph = builder.compile(checkpointer=memory)

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "passenger_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": 1,
    }
}
_printed = set()
while True:
    question = input("YOU : ")
    events = part_1_graph.invoke(
        {"messages": ("user", question)}, config, stream_mode="values"
    )
    # for event in events:
    #     _print_event(event, _printed)
    print("_________________FINAL_____________________")
    print("\n\n\n",events["messages"][-1].content)