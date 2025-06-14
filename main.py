# main.py

from langchain_mistralai import ChatMistralAI
from langchain.chains import RetrievalQA
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
from pdf_utils import extract_numbered_questions, create_vector_store_from_pdf
import os

load_dotenv()

def get_llm():
    try:
        return ChatMistralAI(
            model="mistral-small",  # or "mistral-medium", "mistral-large"
            temperature=0.4
        )
    except Exception as e:
        print("❌ Failed to initialize MistralAI LLM:", e)
        exit()

def ask_tutor(llm, prompt):
    messages = [
        SystemMessage(content="You are a helpful and patient AI tutor. Explain concepts clearly."),
        HumanMessage(content=prompt)
    ]
    try:
        return llm.invoke(messages).content
    except Exception as e:
        return f"⚠️ Error: {e}"

def ask_question_from_pdf(llm, vector_store, query):
    try:
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=vector_store.as_retriever())
        return qa.run(query)
    except Exception as e:
        return f"⚠️ Error with PDF QA: {e}"

if __name__ == "__main__":
    llm = get_llm()
    pdf_path = "chapter1.pdf"  # replace with your own file
    print("📄 Loading and processing PDF...")

    question_dict = extract_numbered_questions(pdf_path)
    vector_store = create_vector_store_from_pdf(pdf_path)

    print(f"✅ Found {len(question_dict)} questions in PDF.")
    print("Ask by typing:")
    print(" - a number (e.g. 5)")
    print(" - multiple (e.g. 2,4,7)")
    print(" - 'all' to answer every question")
    print(" - freeform questions also supported\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        if user_input.lower() == "all":
            for num, q in question_dict.items():
                print(f"\nQ{num}: {q}")
                print("Answer:", ask_question_from_pdf(llm, vector_store, q))
        elif user_input.replace(",", "").replace(" ", "").isdigit():
            try:
                nums = [int(n.strip()) for n in user_input.split(",")]
                for num in nums:
                    question = question_dict.get(num)
                    if question:
                        print(f"\nQ{num}: {question}")
                        print("Answer:", ask_question_from_pdf(llm, vector_store, question))
                    else:
                        print(f"❌ Question {num} not found.")
            except ValueError:
                print("⚠️ Invalid numbers.")
        else:
            # Freeform non-numbered question
            print("Tutor:", ask_question_from_pdf(llm, vector_store, user_input))
