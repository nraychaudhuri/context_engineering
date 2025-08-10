from typing import TypedDict
import uuid
import os
import sqlite3
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command


class State(TypedDict):
    text_1: str
    text_2: str


def human_node_1(state: State):
    print("calling interrupt for text_1")
    value = interrupt("What is text_1?")
    print("calling interrupt for text_1 1")

    return {"text_1": value}


def human_node_2(state: State):
    value = interrupt("What is text_2?")
    return {"text_2": value}


graph_builder = StateGraph(State)
graph_builder.add_node("human_node_1", human_node_1)
graph_builder.add_node("human_node_2", human_node_2)

# Add both nodes in parallel from START
graph_builder.add_edge(START, "human_node_1")
graph_builder.add_edge(START, "human_node_2")

# SQLite configuration - creates a local database file
DB_PATH = "checkpoints.db"

# Create SQLite connection and checkpointer
# Note: We need to keep the connection alive for the checkpointer to work
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Compile the graph with checkpointer
graph = graph_builder.compile(checkpointer=checkpointer)

thread_id = str(uuid.uuid4())

config: RunnableConfig = {"configurable": {"thread_id": thread_id}}


def start_workflow():
    """Start the workflow and return when interrupts occur"""
    print("Starting graph execution...")
    result = graph.invoke(input={}, config=config)

    # Check if workflow completed or was interrupted
    state = graph.get_state(config)
    if state.interrupts:
        print(f"Workflow paused with {len(state.interrupts)} interrupts")
        return None  # Indicates workflow was interrupted
    else:
        print("Workflow completed without interrupts")
        return result


def check_interrupts():
    """Check for any pending interrupts"""
    state = graph.get_state(config)
    print(f"Current state: {state.values}")
    print(f"Number of interrupts: {len(state.interrupts)}")
    return state.interrupts


def handle_interrupts(interrupts):
    """Handle interrupts by collecting user input"""
    resume_map = {}
    for interrupt_obj in interrupts:
        print(f"\nInterrupt ID: {interrupt_obj.interrupt_id}")
        print(f"Prompt: {interrupt_obj.value}")

        # Get actual user input
        user_input = input(f"Please provide input for '{interrupt_obj.value}': ")
        resume_map[interrupt_obj.interrupt_id] = user_input

    return resume_map


def resume_workflow(resume_map):
    """Resume the workflow with user inputs"""
    print(f"\nResuming with user inputs: {resume_map}")
    final_result = graph.invoke(Command(resume=resume_map), config=config)
    print(f"Final result: {final_result}")
    return final_result


# Main execution flow
if __name__ == "__main__":
    print(f"Thread ID: {thread_id}")
    print(f"Using SQLite database: {DB_PATH}")

    # Start the workflow - this will hit interrupts and save checkpoint
    result = start_workflow()

    if result is None:  # Workflow was interrupted
        print(f"\n{'='*50}")
        print("WORKFLOW PAUSED AT INTERRUPTS")
        print(f"{'='*50}")
        print(f"Thread ID: {thread_id}")
        print(f"Database: {DB_PATH}")
        print("\nTo resume this workflow, run:")
        print(f"python resume_workflow.py {thread_id}")
        print(f"{'='*50}")
    else:
        print("Workflow completed:", result)
