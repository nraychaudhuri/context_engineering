#!/usr/bin/env python3
"""
Resume script for human-in-the-loop workflows stored in SQLite.
This script can be run separately to resume a paused workflow.

Usage:
    python resume_workflow.py <thread_id>
"""

import sys
import os
import sqlite3
from typing import TypedDict
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command

# SQLite configuration
DB_PATH = "checkpoints.db"


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


def create_graph():
    """Create and compile the graph with SQLite checkpointer"""
    graph_builder = StateGraph(State)
    graph_builder.add_node("human_node_1", human_node_1)
    graph_builder.add_node("human_node_2", human_node_2)

    # Add both nodes in parallel from START
    graph_builder.add_edge(START, "human_node_1")
    graph_builder.add_edge(START, "human_node_2")

    # Create SQLite connection and checkpointer
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return graph_builder.compile(checkpointer=checkpointer)


def resume_workflow(thread_id: str):
    """Resume a workflow by thread_id"""
    graph = create_graph()
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    # Check current state
    state = graph.get_state(config)

    if not state:
        print(f"No workflow found for thread_id: {thread_id}")
        return

    print(f"Found workflow for thread_id: {thread_id}")
    print(f"Current state: {state.values}")
    print(f"Number of interrupts: {len(state.interrupts)}")

    if not state.interrupts:
        print("No pending interrupts found")
        return

    # Handle interrupts
    resume_map = {}
    for interrupt_obj in state.interrupts:
        print(f"\nInterrupt ID: {interrupt_obj.interrupt_id}")
        print(f"Prompt: {interrupt_obj.value}")

        # Get actual user input
        user_input = input(f"Please provide input for '{interrupt_obj.value}': ")
        resume_map[interrupt_obj.interrupt_id] = user_input

    # Resume workflow
    print(f"\nResuming with user inputs: {resume_map}")
    final_result = graph.invoke(Command(resume=resume_map), config=config)
    print(f"Final result: {final_result}")


def list_workflows():
    """List all workflows in SQLite database"""
    try:
        if not os.path.exists(DB_PATH):
            print(f"No database file found at {DB_PATH}")
            return

        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()

        # Check if the checkpoints table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints';"
        )
        if not cursor.fetchone():
            print("No checkpoints table found in database")
            conn.close()
            return

        # Get all unique thread_ids from checkpoints
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints;")
        rows = cursor.fetchall()

        if rows:
            print(f"Found {len(rows)} workflow threads in {DB_PATH}:")
            for row in rows:
                thread_id = row[0]
                print(f"  - {thread_id}")
        else:
            print("No workflows found in database")

        conn.close()

    except Exception as e:
        print(f"Error listing workflows: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python resume_workflow.py <thread_id>  # Resume specific workflow")
        print("  python resume_workflow.py --list       # List all workflows")
        sys.exit(1)

    arg = sys.argv[1]

    if arg == "--list":
        list_workflows()
    else:
        thread_id = arg
        resume_workflow(thread_id)
