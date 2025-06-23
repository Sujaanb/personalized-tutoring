# main.py
from langchain_mistralai import ChatMistralAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()  # Load .env variables (especially for API keys)

# ---- Initialize the Mistral LLM ----
def get_llm():
    try:
        return ChatMistralAI(
            model="mistral-small",  # Options: mistral-small, mistral-medium, mistral-large
            temperature=0.4
        )
    except Exception as e:
        print("❌ Failed to initialize MistralAI LLM:", e)
        exit(1)

# ---- Ask question directly without context ----
def ask_llm_direct(llm, user_input):
    messages = [
        SystemMessage(content="You are a helpful and patient AI tutor. Explain concepts clearly."),
        HumanMessage(content=user_input)
    ]
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"⚠️ Error generating response: {e}"

# ---- Command-line interface ----
if __name__ == "__main__":
    llm = get_llm()
    print("🎓 AI Tutor (Mistral) Ready! Type your questions below (type 'exit' to quit):\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("👋 Goodbye!")
            break

        answer = ask_llm_direct(llm, user_input)
        print("Tutor:", answer)
