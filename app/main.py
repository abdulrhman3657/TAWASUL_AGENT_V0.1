# This file is the command-line chat app.

import os # os is used to read environment variables (like OPENAI_MODEL).

# build_agent is your factory function from app/agent.py that:
# creates the OpenAI chat model,
# registers all the tools (save_text, call_api, send_email, search_docs),
# wires memory (ConversationBufferMemory + MessagesPlaceholder),
# returns a LangChain AgentExecutor object.
from .agent import build_agent 

print("Loaded key:", os.getenv("OPENAI_API_KEY"))


def main():
    # Reads OPENAI_MODEL from env, defaulting to "gpt-4o-mini":
    # checks for an environment variable called OPENAI_MODEL.
    # If it exists, it uses that (e.g. "gpt-4o" or your preferred model).
    # If not, it defaults to "gpt-4o-mini".
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Builds the agent:
    # Inside build_agent:
    # ChatOpenAI(model=model, temperature=0.2) -> the LLM
    # Create StructuredTools
    # Initialize an AgentExecutor
    agent = build_agent(model=model)

    print("Agent ready. Type 'exit' to quit.")
    
    # Runs an input loop
    # use python -m app.main
    while True:
        try:
            user = input("\nYou: ").strip()
            if user.lower() in {"exit", "quit"}:
                break
            result = agent.invoke({"input": user})
            reply = result.get("output", str(result))
            print("Agent:", reply)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
