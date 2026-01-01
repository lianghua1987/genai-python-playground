import operator
from dotenv import load_dotenv
from typing import Annotated, List, Sequence, TypedDict
from dotenv import dotenv_values
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI  # or your favourite model
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph

import os
# ── 1. Define Tools (example) ───────────────────────────────────────
from langchain_community.tools.tavily_search import TavilySearchResults

# config = dotenv_values("../../.env")
# if gemini_key := config.get("GEMINI_API_KEY"):
#     os.environ["GEMINI_API_KEY"] = gemini_key
# else:
#     raise ValueError("GEMINI_API_KEY not found in .env")
#
# if tavily_key := config.get("TAVILY_API_KEY"):
#     os.environ["TAVILY_API_KEY"] = tavily_key
# else:
#     raise ValueError("TAVILY_API_KEY not found in .env")
load_dotenv()

tools = [TavilySearchResults(max_results=1)]
tool_node = ToolNode(tools=tools)

# ── 2. State ───────────────────────────────────────────────────────────
class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[tuple], operator.add]
    response: str


# ── 3. LLM & Prompts ──────────────────────────────────────────────────
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # or gpt-4o, claude-3.5-sonnet, etc.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    max_output_tokens=2048,
)  # or gpt-4o, claude-3.5-sonnet, etc.
llm_with_tools = llm.bind_tools(tools)


# Planner
planner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert planner.
Given the user objective, come up with a simple step-by-step plan.
Each step should be concise and independent.
Return ONLY the plan as a numbered list inside <plan>...</plan> tags.

Objective: {objective}"""),
    ("human", "Current objective: {objective}"),
])

planner = planner_prompt | llm


# Replanner
replanner_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a replanner.
You have this objective: {objective}

Previous plan:
{plan}

Steps already executed with results:
{past_steps}

Now decide:
- If you already have enough information → respond with the final answer inside <final_answer>...</final_answer>
- Otherwise → create a new, updated plan (or continue the old one).
Return ONLY the updated plan as a numbered list inside <plan>...</plan> tags or final answer inside <final_answer>...</final_answer> tags.
Never explain outside tags."""),
    MessagesPlaceholder("messages"),
])

replanner = replanner_prompt | llm


def parse_plan(text: str) -> List[str]:
    """Very simple parser — improve as needed"""
    if "<plan>" not in text:
        return []
    plan_text = text.split("<plan>")[1].split("</plan>")[0].strip()
    steps = [s.strip() for s in plan_text.split("\n") if s.strip() and s.strip()[0].isdigit()]
    return [s.split(".", 1)[1].strip() if "." in s else s for s in steps]


def parse_final_answer(text: str) -> str | None:
    if "<final_answer>" in text:
        return text.split("<final_answer>")[1].split("</final_answer>")[0].strip()
    return None


# ── 4. Nodes ───────────────────────────────────────────────────────────
def plan_step(state: PlanExecuteState):
    plan_result = planner.invoke({"objective": state["input"]})
    plan_str = plan_result.content
    plan_list = parse_plan(plan_str)
    print("Generated plan:", plan_list)
    return {"plan": plan_list}


def execute_step(state: PlanExecuteState):
    """Execute the next step in the plan"""
    if not state["plan"]:
        return {"past_steps": [("No more steps", "Plan empty")]}

    next_step = state["plan"][0]
    print(f"\nExecuting step: {next_step}")

    # Simple execution: just ask the model to do the step using tools
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Execute ONLY this step: {next_step}
Use tools when needed. Be concise."""),
        MessagesPlaceholder("messages"),
        ("human", "Current step to execute: {step}"),
    ])

    chain = agent_prompt | llm_with_tools

    result = chain.invoke({
        "messages": [HumanMessage(content=state["input"])],  # context
        "step": next_step
    })

    if isinstance(result, AIMessage) and result.tool_calls:
        return {
            "past_steps": [(next_step, "Tool calls needed → will execute in next step")],
            # We will handle tool execution in separate node
        }
    else:
        return {
            "past_steps": [(next_step, result.content)],
            "plan": state["plan"][1:]  # pop first step
        }


def execute_tools(state: PlanExecuteState):
    """Run tools if needed (alternative flow)"""
    # You can expand this if you want full ReAct inside each step
    # For simplicity we mostly rely on model doing tool calls in execute_step
    return state  # stub — expand if needed


def should_end_or_replan(state: PlanExecuteState):
    if not state["plan"]:
        return "replan"

    last_result = state["past_steps"][-1][1] if state["past_steps"] else ""
    if "enough" in last_result.lower() or len(state["plan"]) == 0:
        return "replan"  # force replan to check for final answer

    return "agent"  # continue executing


def replan_step(state: PlanExecuteState):
    past_steps_str = "\n".join([f"- {s}: {r}" for s, r in state["past_steps"]])

    replan_result = replanner.invoke({
        "objective": state["input"],
        "plan": "\n".join(f"{i+1}. {p}" for i, p in enumerate(state["plan"])),
        "past_steps": past_steps_str,
        "messages": [HumanMessage(content=state["input"])]
    })

    content = replan_result.content
    final = parse_final_answer(content)

    if final is not None:
        return {"response": final}

    new_plan = parse_plan(content)
    if not new_plan:
        return {"response": "Could not generate new plan. Giving up."}

    return {
        "plan": new_plan,
        "past_steps": []  # optional: reset or keep history
    }


# ── 5. Build Graph ─────────────────────────────────────────────────────
workflow = StateGraph(PlanExecuteState)

workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("tools", tool_node)           # optional
workflow.add_node("replan", replan_step)

workflow.set_entry_point("planner")

workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")           # most common loop
workflow.add_edge("tools", "agent")            # if you use separate tool node

workflow.add_conditional_edges(
    "replan",
    lambda s: "agent" if s.get("plan") and not s.get("response") else END,
    {"agent": "agent", END: END}
)

app = workflow.compile()  # add checkpointer=MemorySaver() for memory


# ── 6. Run example ─────────────────────────────────────────────────────
if __name__ == "__main__":
    result = app.invoke({
        "input": "Who wins the US open recently?",
        "plan": [],
        "past_steps": [],
        "response": ""
    })

    print("\n" + "="*60)
    print("Final Answer:")
    print(result["response"])