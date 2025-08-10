from typing import TypedDict
import uuid
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command


class State(TypedDict):
    text_1: str
    text_2: str


def human_node_1(state: State):
    print("caling interrupt for text_1")
    value = interrupt("text_to_revise 1")
    print("caling interrupt for text_1 1")

    return {"text_1": value}


def human_node_2(state: State):
    value = interrupt("text_to_revise 2")
    return {"text_2": value}


graph_builder = StateGraph(State)
graph_builder.add_node("human_node_1", human_node_1)
graph_builder.add_node("human_node_2", human_node_2)

# Add both nodes in parallel from START
graph_builder.add_edge(START, "human_node_1")
graph_builder.add_edge(START, "human_node_2")

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

thread_id = str(uuid.uuid4())

config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
# result = graph.invoke(
#     {"text_1": "original text 1", "text_2": "original text 2"}, config=config
# )
result = graph.invoke(input={}, config=config)

# # Resume with mapping of interrupt IDs to values
# resume_map = {
#     i.interrupt_id: f"human input for prompt {i.value}"
#     for i in graph.get_state(config).interrupts
# }
# print(graph.invoke(Command(resume=resume_map), config=config))
# > {'text_1': 'edited text for original text 1', 'text_2': 'edited text for original text 2'}
