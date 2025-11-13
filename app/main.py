
import os
from .agent import build_agent

import os
print("Loaded key:", os.getenv("OPENAI_API_KEY"))


def main():
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    agent = build_agent(model=model)
    print("Agent ready. Type 'exit' to quit.")
    while True:
        try:
            user = input("\nYou: ").strip()
            if user.lower() in {"exit", "quit"}:
                break
            result = agent.run(user)
            print("Agent:", result)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
