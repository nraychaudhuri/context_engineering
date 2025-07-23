import os
import openai
import tempfile
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.tools import tool
import json
from tools.exa_tool import exa_search
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

load_dotenv()

TODO_FILE = os.path.join(tempfile.gettempdir(), "todo.md")


# -------------------- Agent State --------------------
class AgentState(TypedDict):
    task: str
    todo_md: str
    step_results: List[str]
    step_count: int


# -------------------- Tools --------------------


@tool
def search_web(query: str) -> list[dict[str, str]]:
    """Use this to search the web for a given query.

    Args:
        query: The search query.
    Returns:
        A list of search results, each containing a title, URL and text.
    """
    results = exa_search(query)
    return results


@tool
def summarize_text(text: str) -> str:
    """Use this to summarize a given text.

    Args:
        text: The text to summarize.
    Returns:
        A summary of the input text.
    """

    messages = [
        {
            "role": "system",
            "content": "Summarize this text in 3 concise bullet points.",
        },
        {"role": "user", "content": text},
    ]
    model = ChatOpenAI(model="gpt-4.1-mini", temperature=None)
    response = model.invoke(messages)

    return response.choices[0].message["content"]


# -------------------- Tool Schemas --------------------
tool_definitions = [
    {
        "name": "search_web",
        "description": "Search the web for a given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "summarize_text",
        "description": "Summarize the input text.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The text to summarize."}
            },
            "required": ["text"],
        },
    },
]


# -------------------- TODO File Management --------------------
def load_todo():
    return open(TODO_FILE).read() if os.path.exists(TODO_FILE) else ""


def save_todo(todo: str):
    with open(TODO_FILE, "w") as f:
        f.write(todo)


# -------------------- Initialization --------------------
def initialize_agent(task: str) -> AgentState:
    messages = [
        {
            "role": "system",
            "content": "Break this task into a TODO list in markdown format (3–5 steps).",
        },
        {"role": "user", "content": task},
    ]
    model = ChatOpenAI(model="gpt-4.1-mini", temperature=None)
    response = model.invoke(messages)
    print(">>>>> message ", response)
    todo_md = response.content
    save_todo(todo_md)

    return {"task": task, "todo_md": todo_md, "step_results": [], "step_count": 0}


# -------------------- Perform Step (with Function Calling) --------------------
def perform_next_step(state: AgentState) -> AgentState:
    messages = [
        {
            "role": "system",
            "content": "Pick the next step from the TODO list and call the right tool.",
        },
        {
            "role": "user",
            "content": f"Task: {state['task']}\n\nTODO.md:\n{state['todo_md']}\n\nPrevious Results:\n"
            + "\n".join(state["step_results"]),
        },
    ]
    model = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=None,
        # tools=tool_definitions,
        # tool_choice="auto",
    )
    model = model.bind_tools(tools=tool_definitions, tool_choice="auto")
    response = model.invoke(messages)
    print(">>>>> response111 ", response)
    tool_call = response.tool_calls[0]
    print(">>>>> response111 ", tool_call)
    if not tool_call:
        state["step_results"].append("No tool selected.")
        return {**state, "step_count": state["step_count"] + 1}

    name = tool_call["name"]
    # args = json.loads(tool_call["args"])
    args = tool_call["args"]

    if name == "search_web":
        result = search_web(args["query"])
    elif name == "summarize_text":
        result = summarize_text(args["text"])
    else:
        result = f"Unknown tool: {name}"

    state["step_results"].append(f"[{name}] {result}")
    return {**state, "step_count": state["step_count"] + 1}


# -------------------- Update TODO Recitation --------------------
def update_todo(state: AgentState) -> AgentState:
    messages = [
        {
            "role": "system",
            "content": "Update the TODO.md list by checking off what’s completed, based on the latest results.",
        },
        {
            "role": "user",
            "content": f"Task: {state['task']}\n\nPrevious TODO.md:\n{state['todo_md']}\n\nStep Results:\n"
            + "\n".join(state["step_results"]),
        },
    ]

    model = ChatOpenAI(model="gpt-4.1-mini", temperature=None)
    response = model.invoke(messages)
    updated_todo = response.content

    save_todo(updated_todo)

    return {**state, "todo_md": updated_todo}


# -------------------- Loop Control --------------------
def is_done(state: AgentState) -> str:
    if "[ ]" in state["todo_md"] and state["step_count"] < 10:
        return "continue"
    return END


# -------------------- LangGraph Setup --------------------
def build_agent():
    builder = StateGraph(AgentState)
    builder.add_node("perform_next_step", perform_next_step)
    builder.add_node("update_todo", update_todo)
    builder.set_entry_point("perform_next_step")
    builder.add_edge("perform_next_step", "update_todo")
    builder.add_conditional_edges(
        "update_todo", is_done, {"continue": "perform_next_step", END: END}
    )
    return builder.compile()


# -------------------- Run Agent --------------------
if __name__ == "__main__":
    user_task = "Research 3 AI startups and summarize their impact."
    state = initialize_agent(user_task)
    agent = build_agent()

    for step in agent.stream(state, stream_mode="values"):
        print(f"\n🔁 Step {step['step_count']} — TODO.md:\n{step['todo_md']}\n")
        print("📝 Step Results:\n", "\n---\n".join(step["step_results"]))
