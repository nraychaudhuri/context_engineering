"""
Hybrid Reflexion over LangGraph's prebuilt ReAct agent
=====================================================

This example starts from LangGraph's prebuilt `create_react_agent` and layers a
**hybrid reflection loop** on top:

    ReAct (actor) → Evaluate (deterministic PASS/FAIL) →
    [optional Critic LLM] → Reflect (distill 1–4 concrete fixes) → retry

It demonstrates how to:
  • Inject short reflection notes via a **dynamic prompt** for `create_react_agent`.
  • Keep a tiny episodic memory (`reflections`) and cap it to k tips.
  • Use a deterministic evaluator for reliability, with an optional LLM critic
    to enrich feedback when the evaluator is coarse or ambiguous.
  • Wire everything as a `StateGraph` with conditional edges and a checkpointer.

This file uses an essay task to keep the example self-contained. Swap the
`Evaluator` with your web evaluator (CSS/XPath checks, numeric thresholds, etc.)
and it becomes a robust web agent loop.

Dependencies (example):
    pip install -U langgraph langchain langchain-openai tavily-python
Set env var:  OPENAI_API_KEY

Run:
    python hybrid_reflexion_langgraph.py


Langgraph References:

https://langchain-ai.github.io/langgraph/tutorials/reflection/reflection/?utm_source=chatgpt.com
"""

from __future__ import annotations
from dotenv import load_dotenv

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Example tool: Tavily search (or replace with your own tools)
from langchain_tavily import TavilySearch


load_dotenv()
# -----------------------------
# Config knobs
# -----------------------------
MAX_ATTEMPTS = 3
K_REFLECTIONS = 2
CRITIC_ON_FAIL = True  # set False to skip the critic step
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0


# -----------------------------
# State definition
# -----------------------------
class ReflexState(AgentState, total=False):

    reflections: List[str]
    attempt: int
    verdict: Literal["pass", "fail", "give_up"]
    reasons: List[str]
    evidence: Dict[str, Any]
    critic: Optional[str]


# -----------------------------
# Dynamic prompt that injects reflections
# -----------------------------
def dynamic_prompt(state: ReflexState, config: RunnableConfig) -> List[AnyMessage]:
    notes = state.get("reflections", [])[-K_REFLECTIONS:]
    base = (
        "You are a careful ReAct agent. Think step by step, verify assumptions, "
        "and follow the correction notes strictly when present."
    )
    if notes:
        base += "\n\nCorrection notes to follow:\n" + "\n".join(f"- {n}" for n in notes)
    return [SystemMessage(content=base)] + state["messages"]


# -----------------------------
# LLMs
# -----------------------------
# You can also use `init_chat_model` to configure temperature, etc.
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=TEMPERATURE)
# llm = init_chat_model(MODEL_NAME, temperature=TEMPERATURE)
critic_llm = llm  # reuse; you may choose a larger/cheaper model
reflect_llm = llm


# -----------------------------
# Tools (example)
# -----------------------------
search_tool = TavilySearch(max_results=5)
# search_tool = TavilySearchResults(api_wrapper=search_api, max_results=5)
TOOLS = [search_tool]


# -----------------------------
# Build the ReAct actor using the prebuilt agent
# -----------------------------
checkpointer = InMemorySaver()
react_agent = create_react_agent(
    model=llm,
    tools=TOOLS,
    prompt=dynamic_prompt,  # dynamic system prompt built from state
    checkpointer=checkpointer,
)


# -----------------------------
# Deterministic evaluator for the essay task (replace in your app)
# -----------------------------
@dataclass
class EvalResult:
    passed: bool
    reasons: List[str]
    score: Optional[float] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


def _count_paragraphs(text: str) -> int:
    paras = [p for p in text.split("\n") if p.strip()]
    return len(paras)


def evaluate_essay(messages: List[AnyMessage]) -> EvalResult:
    """Toy evaluator: PASS if the final AI message meets simple rules.

    Rules (illustrative):
      1) At least 5 paragraphs
      2) Contains a thesis-ish sentence marker ("In conclusion" or "This essay" or "Therefore,")
      3) Mentions at least one source indicator ("http" or "[1]")
    """
    # Find last AI message
    last_ai = None
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            last_ai = m
            break
    if last_ai is None:
        return EvalResult(False, ["No AI answer produced."])

    text = last_ai.content if isinstance(last_ai.content, str) else str(last_ai.content)
    paras = _count_paragraphs(text)
    has_thesis = any(
        key in text.lower()
        for key in [
            "in conclusion",
            "this essay",
            "therefore,",
            "overall,",
            "to conclude",
        ]
    )
    has_source = ("http" in text) or ("[1]" in text)

    reasons = []
    if paras < 5:
        reasons.append(f"Not enough paragraphs: {paras} < 5")
    if not has_thesis:
        reasons.append("Missing explicit thesis/conclusion marker.")
    if not has_source:
        reasons.append("No source/citation-like marker detected.")

    passed = len(reasons) == 0
    return EvalResult(
        passed=passed,
        reasons=["All checks passed."] if passed else reasons,
        score=float(paras >= 5) + float(has_thesis) + float(has_source),
        evidence={
            "paragraphs": paras,
            "has_thesis": has_thesis,
            "has_source": has_source,
        },
    )


# -----------------------------
# Optional critic node (LLM)
# -----------------------------
critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a tough but helpful critic. Given an answer and the evaluator's"
            " failure reasons, provide a short critique that explains likely root causes"
            " and missing steps. Keep it under 120 words.",
        ),
        ("human", "Evaluator reasons:\n{reasons}\n\nAnswer:\n{answer}"),
    ]
)
critic_chain = critic_prompt | critic_llm


# -----------------------------
# Reflection node: distill 1–4 concrete fixes
# -----------------------------
reflect_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You write SHORT, ACTIONABLE corrections for an agent. Based on the"
            " evaluator reasons and optional critic feedback, output 1–4 bullet points"
            " that are concrete (order, selectors/sections, thresholds, checks)."
            " Each bullet <= 120 characters.",
        ),
        (
            "human",
            "Task: {task}\n\nEvaluator reasons:\n{reasons}\n\nCritic (optional):\n{critic}\n",
        ),
    ]
)
reflect_chain = reflect_prompt | reflect_llm


# -----------------------------
# Graph node functions
# -----------------------------
def node_react(state: ReflexState, config: RunnableConfig) -> ReflexState:
    """Run the prebuilt ReAct agent with current state (messages + reflections)."""
    # Provide thread_id to enable checkpointer persistence
    result = react_agent.invoke(state, config)
    print(">>>>> react_agent result:", result)
    # `result` contains updated messages (and more if structured output was used)
    return {"messages": result["messages"]}


def node_evaluate(state: ReflexState) -> ReflexState:
    attempt = state.get("attempt", 1)
    eval_res = evaluate_essay(state["messages"])

    verdict: Literal["pass", "fail", "give_up"]
    if eval_res.passed:
        verdict = "pass"
    else:
        verdict = "fail" if attempt < MAX_ATTEMPTS else "give_up"

    return {
        "verdict": verdict,
        "reasons": eval_res.reasons,
        "evidence": eval_res.evidence,
    }


def node_critic(state: ReflexState) -> ReflexState:
    # Build last AI answer string
    last_ai_txt = ""
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            last_ai_txt = m.content if isinstance(m.content, str) else str(m.content)
            break
    msg = critic_chain.invoke(
        {
            "reasons": "\n".join(state.get("reasons", [])),
            "answer": last_ai_txt,
        }
    )
    print(">>>>> critic result:", msg.content)
    return {"critic": str(msg.content)}


def node_reflect(state: ReflexState) -> ReflexState:
    critic_txt = state.get("critic", "") if CRITIC_ON_FAIL else ""
    task_hint = (
        "Write a 5-paragraph essay with a clear thesis and at least one citation."
    )
    msg = reflect_chain.invoke(
        {
            "task": task_hint,
            "reasons": "\n".join(state.get("reasons", [])),
            "critic": critic_txt,
        }
    )
    # Parse bullets
    tips = [
        line.strip("- ").strip()
        for line in str(msg.content).splitlines()
        if line.strip()
    ]
    tips = tips[:4]

    prev = state.get("reflections", [])
    new_reflections = (prev + tips)[-max(K_REFLECTIONS, len(tips)) :]
    print(">>>>> reflection tips: ", new_reflections)
    return {"reflections": new_reflections, "attempt": state.get("attempt", 1) + 1}


# Routing helpers --------------------------------------------------------------


def route_after_eval(state: ReflexState) -> Literal["pass", "fail", "give_up"]:
    return state["verdict"]


# -----------------------------
# Build and compile the graph
# -----------------------------
builder = StateGraph(ReflexState)

builder.add_node("react", node_react)
builder.add_node("evaluate", node_evaluate)
if CRITIC_ON_FAIL:
    builder.add_node("critic", node_critic)
builder.add_node("reflect", node_reflect)

builder.add_edge(START, "react")
builder.add_edge("react", "evaluate")

if CRITIC_ON_FAIL:
    # evaluate → pass/end ; fail→critic→reflect ; give_up→end
    builder.add_conditional_edges(
        "evaluate",
        route_after_eval,
        {
            "pass": END,
            "fail": "critic",
            "give_up": END,
        },
    )
    builder.add_edge("critic", "reflect")
else:
    builder.add_conditional_edges(
        "evaluate",
        route_after_eval,
        {
            "pass": END,
            "fail": "reflect",
            "give_up": END,
        },
    )

builder.add_edge("reflect", "react")

app = builder.compile(checkpointer=checkpointer)


# -----------------------------
# Demo
# -----------------------------
def _demo() -> None:
    print("\n=== Hybrid Reflexion over ReAct demo ===\n")
    user_request = HumanMessage(
        content=(
            "Write an essay on why the AGI is important for the future of humanity. Make sure to include include sources and a clear thesis."
            "Return a well-structured answer."
        )
    )
    # initial state
    state: ReflexState = {
        "messages": [user_request],
        "reflections": [],
        "attempt": 1,
    }

    cfg: RunnableConfig = {"configurable": {"thread_id": "demo-thread-1"}}

    result = app.invoke(state, cfg)

    print("\n--- Attempts:", result.get("attempt", 1))
    print("--- Verdict:", result.get("verdict"))
    if result.get("reasons"):
        print("--- Reasons:")
        for r in result["reasons"]:
            print("  -", r)
    if result.get("critic"):
        print("--- Critic (short):", result["critic"][:200], "...")
    if result.get("reflections"):
        print("--- Final reflections loaded into next run:")
        for tip in result["reflections"]:
            print("  -", tip)

    # Print the final AI message
    last_ai = None
    for m in reversed(result["messages"]):
        if isinstance(m, AIMessage):
            last_ai = m
            break
    if last_ai:
        print("\n=== Final Answer ===\n")
        text = (
            last_ai.content
            if isinstance(last_ai.content, str)
            else str(last_ai.content)
        )
        print(text)


if __name__ == "__main__":
    _demo()
