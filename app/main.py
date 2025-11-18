# app/main.py
# This file is the command-line chat app.

import os
from .agent import build_agent
from .tools import save_text_tool
from .fallback_detector import is_semantic_fallback


print("Loaded key:", os.getenv("OPENAI_API_KEY"))


def main():
    """
    Interactive CLI chatbot.

    Features:
    - Uses the same agent as Streamlit + FastAPI.
    - Embedding-based fallback detection → auto-log FAQ candidates.
    - Never crashes the CLI on LLM/tool errors.
    """

    # Pick model from environment or default
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    # Build the agent (LLM, tools, memory, system prompt)
    agent = build_agent(model=model)

    print("Agent ready. Type 'exit' to quit.")

    # REPL loop
    while True:
        try:
            user = input("\nYou: ").strip()
            if user.lower() in {"exit", "quit"}:
                break

            # Run agent
            result = agent.invoke({"input": user})
            reply = result.get("output", str(result))
            print("Agent:", reply)

            # --- Semantic fallback detection → FAQ auto-logging ---
            try:
                is_fb, score = is_semantic_fallback(reply)
                if is_fb:
                    # Prevent logging private data (very basic filter)
                    has_email = "@" in user
                    number_digits = "".join(c for c in user if c.isdigit())
                    is_long_number = len(number_digits) >= 6

                    if not has_email and not is_long_number:
                        save_text_tool(
                            text=user,
                            tag="faq_candidate",
                            meta={
                                "source": "cli",
                                "similarity": score,
                            },
                        )
                        print("(Logged as FAQ candidate)")
            except Exception as e:
                print("Warning: FAQ logging failed:", e)

        except KeyboardInterrupt:
            break

        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()


# To run:
# python -m app.main