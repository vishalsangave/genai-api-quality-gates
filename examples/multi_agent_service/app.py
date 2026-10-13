from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

class ChatResponse(BaseModel):
    response: str
    agent_name: str
    tool_calls: list = []
    trace: dict | None = None  # picked up by DriftGate's trace capture

# --- Mock Agents ---

def summarizer_agent(text: str) -> str:
    greetings = ["Here's the summary:", "To summarize:", "In short:"]
    # Simulate non-determinism
    greeting = random.choice(greetings)
    return f"{greeting} The user wants to summarize the text '{text}'."

def translator_agent(text: str) -> str:
    greetings = ["Translation:", "In French:", "Voici la traduction:"]
    # Simulate non-determinism
    greeting = random.choice(greetings)
    return f"{greeting} The user wants to translate the text '{text}'."

def math_agent(text: str) -> tuple[str, list]:
    # Simulate tool use
    tool_calls = [{"tool_name": "calculator", "tool_input": text}]
    try:
        # A very basic, unsafe 'eval' to simulate calculation
        result = eval(text.replace("calculate", "").strip())
        response_text = f"The result is {result}."
    except Exception:
        response_text = "I couldn't calculate that."
    return response_text, tool_calls

# --- Orchestrator ---

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    prompt = request.prompt.lower()
    tool_calls = []

    if "summarize" in prompt:
        agent_name = "SummarizerAgent"
        response_text = summarizer_agent(request.prompt)
    elif "translate" in prompt:
        agent_name = "TranslatorAgent"
        response_text = translator_agent(request.prompt)
    elif "calculate" in prompt:
        agent_name = "MathAgent"
        response_text, tool_calls = math_agent(request.prompt)
    else:
        agent_name = "DefaultAgent"
        response_text = "I'm not sure how to handle that. I can summarize, translate, or calculate."

    trace = {
        "task": request.prompt,
        "steps": [
            {"agent": "Orchestrator", "input": {"prompt": request.prompt}, "handoff_to": agent_name},
            {
                "agent": agent_name,
                "output": response_text,
                "tool_calls": [{"name": c["tool_name"], "input": c["tool_input"]} for c in tool_calls],
            },
        ],
        "final_output": response_text,
    }
    return ChatResponse(
        response=response_text, agent_name=agent_name, tool_calls=tool_calls, trace=trace
    )
