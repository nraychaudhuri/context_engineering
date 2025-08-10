import os
import openai
import tempfile
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from langchain_core.tools import tool
import json
from tools.exa_tool import exa_search
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

load_dotenv()

TODO_FILE = os.path.join(tempfile.gettempdir(), "todo.md")


# -------------------- Agent State --------------------
class AgentState(TypedDict):
    task: str
    todo_md: str
    step_results: List[str]
    completed_todos: List[str]  # Track completed tasks
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

    return response.content


def get_next_todo_item(todo_md: str) -> str:
    """Parse the TODO markdown and return the next uncompleted item."""
    lines = todo_md.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("- [ ]") or line.startswith("* [ ]"):
            # Remove the checkbox and return the task description
            return line.replace("- [ ]", "").replace("* [ ]", "").strip()
    return None


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
    # print(">>>>> message ", response)
    todo_md = response.content

    return {
        "task": task,
        "todo_md": todo_md,
        "step_results": [],
        "step_count": 0,
        "completed_todos": [],
    }


def perform_next_step(state: AgentState) -> AgentState:
    # Get the next specific TODO item to work on
    next_item = get_next_todo_item(state["todo_md"])
    print("Next Task: ", next_item)
    if not next_item:
        state["step_results"].append("No more TODO items found.")
        return {**state, "step_count": state["step_count"] + 1}

    # Create ReAct agent with tools
    model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    tools = [search_web, summarize_text]
    react_agent = create_react_agent(model, tools)

    # Prepare input for the ReAct agent
    agent_input = {
        "messages": [
            (
                "human",
                f"""
Context:

Overall Task: {state['task']}

Todo List:
{state['todo_md']}

Previous Results:
{chr(10).join(state["step_results"])}

Complete this specific TODO item: {next_item}. Only focus on this task.
Use the available tools to gather information and complete the task.
""",
            )
        ]
    }

    # Run the ReAct agent
    try:
        result = react_agent.invoke(agent_input)

        # Extract the final message content
        final_message = result["messages"][-1].content

        # Add result to step results
        # aggregated_result = f"Working on: '{next_item}' -> {final_message}"
        state["step_results"].append(final_message)
        state["completed_todos"].append(next_item)

    except Exception as e:
        error_result = f"Error processing '{next_item}': {str(e)}"
        state["step_results"].append(error_result)
        state["completed_todos"].append(next_item)

    return {**state, "step_count": state["step_count"] + 1}


# -------------------- Update TODO Recitation --------------------
def update_todo(state: AgentState) -> AgentState:
    messages = [
        {
            "role": "system",
            "content": "Update the TODO.md list by checking off what’s completed, based on completed todos and step count.",
        },
        {
            "role": "user",
            "content": f"""
Completed todos: {", ".join(state['completed_todos'])}
Step count: {state['step_count']}

Here is the TODO.md:
{state['todo_md']}

""",
        },
    ]

    model = ChatOpenAI(model="gpt-4.1-mini", temperature=None)
    response = model.invoke(messages)
    updated_todo = response.content
    print(updated_todo)
    return {**state, "todo_md": updated_todo}


def is_done(state: AgentState) -> str:
    if "[ ]" in state["todo_md"] and state["step_count"] < 10:
        return "continue"
    return END


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


if __name__ == "__main__":
    user_task = "Research 3 AI startups and summarize their impact."
    state = initialize_agent(user_task)
    agent = build_agent()

    final_state = agent.invoke(state)
    # print(json.dumps(response, indent=2))
    # print(response["step_results"][-1])
    # final_state = None
    # for step in agent.stream(state, stream_mode="values"):
    #     print(f"\n🔁 Step {step['step_count']} — TODO.md:\n{step['todo_md']}\n")
    #     final_state = step

    print("\n" + "=" * 50)
    print("🎯 FINAL RESULTS")
    print("=" * 50)
    print(f"Task: {final_state['task']}")
    print(f"\nCompleted in {final_state['step_count']} steps")
    print(f"\nFinal TODO Status:\n{final_state['todo_md']}")
    print(f"\nLast Step Result:")
    if final_state["step_results"]:
        print(f"{final_state['step_results'][-1]}")
    else:
        print("No step results available")
    print("=" * 50)
