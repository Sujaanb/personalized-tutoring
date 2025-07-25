import json
from langchain_mistralai.chat_models import ChatMistralAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from langchain.schema import HumanMessage


# Import our modules
from config import config
from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from graph_nodes import AgentState, GraphNodes
from upload_service import UploadService

class RAGAssistant:
    """Main RAG Assistant application."""

    def __init__(self):
        # Initialize components
        self.vector_manager = VectorStoreManager(
            api_key=config.mistral_api_key,
            memory_dir=config.chroma_memory_dir,
            kb_dir=config.chroma_kb_dir
        )

        self.llm = ChatMistralAI(
            model=config.model_name,
            api_key=config.mistral_api_key,
            temperature=config.temperature
        )

        self.processor = DocumentProcessor(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )

        self.upload_service = UploadService(self.processor, self.vector_manager)
        self.graph_nodes = GraphNodes(self.llm, self.vector_manager)

        # Initialize the graph
        self.memory_agent = self._create_graph()

        # Reset collections on startup
        # self.vector_manager.reset_collections()

    def _create_graph(self):
        """Create and compile the LangGraph workflow."""
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("retrieve_memory", self.graph_nodes.retrieve_memory_node)
        graph.add_node("retrieve_knowledge", self.graph_nodes.retrieve_knowledge_node)
        graph.add_node("generate_response", self.graph_nodes.generate_response_node)
        graph.add_node("update_memory", self.graph_nodes.update_memory_node)

        # Set entry point and edges
        graph.set_entry_point("retrieve_memory")
        graph.add_edge("retrieve_memory", "retrieve_knowledge")
        graph.add_edge("retrieve_knowledge", "generate_response")
        graph.add_edge("generate_response", "update_memory")
        graph.add_edge("update_memory", END)

        # Generate graph visualization
        app = graph.compile()
        try:
            mermaid_str = app.get_graph().draw_mermaid()
            draw_mermaid_png(
                mermaid_syntax=mermaid_str,
                output_file_path="workflow_graph.png",
                background_color="white",
                padding=20,
            )
            print("📊 Workflow graph saved as workflow_graph.png")
        except Exception as e:
            print(f"⚠️ Could not generate workflow graph: {str(e)}")

        return app

    def process_user_input(self, user_input: str) -> str:
        """Process user input and return response."""
        result = self.memory_agent.invoke({"input": user_input})
        return result.get("response", "Sorry, I couldn't generate a response.")

    def handle_command(self, command: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        command = command.lower().strip()

        if command == "upload":
            result = self.upload_service.upload_documents(config.uploads_directory)
            print(result['message'])
            return True

        elif command == "upload-pdf":
            result = self.upload_service.upload_pdf_files(config.uploads_directory)
            print(result['message'])
            return True

        elif command == "upload-txt":
            result = self.upload_service.upload_txt_files(config.uploads_directory)
            print(result['message'])
            return True

        elif command == "status":
            status = self.vector_manager.get_knowledge_base_status()
            if status['success']:
                print(f"📚 Knowledge base contains {status['total_chunks']} document chunks")
                for file_type, count in status['file_types'].items():
                    print(f"   {'📄' if file_type == 'pdf' else '📝'} {file_type.upper()} chunks: {count}")
            else:
                print(f"📚 Error checking knowledge base: {status.get('error', 'Unknown error')}")
            return True

        elif command == "files":
            file_info = self.upload_service.list_available_files(config.uploads_directory)
            if file_info['exists']:
                print(f"📁 Files in {config.uploads_directory} directory:")
                if file_info['pdf_files']:
                    print(f"   📄 PDF files ({len(file_info['pdf_files'])}):")
                    for filename in file_info['pdf_files']:
                        print(f"      - {filename}")
                if file_info['txt_files']:
                    print(f"   📝 TXT files ({len(file_info['txt_files'])}):")
                    for filename in file_info['txt_files']:
                        print(f"      - {filename}")
                if file_info['total_files'] == 0:
                    print("   (No PDF or TXT files found)")
            else:
                print(file_info['message'])
            return True

        elif command == "quiz":
            self._start_quiz()
            return True

        elif command in ["help", "commands"]:
            self._show_help()
            return True

        return False

    def _show_help(self):
        """Display help information."""
        print("📚 Available commands:")
        print("   'upload'     - Process both .pdf and .txt files from uploads directory")
        print("   'upload-pdf' - Process only .pdf files")
        print("   'upload-txt' - Process only .txt files")
        print("   'status'     - Show knowledge base status")
        print("   'files'      - List files in uploads directory")
        print("   'quiz'       - Start a quiz based on the knowledge base")
        print("   'help'       - Show this help message")
        print("   'exit' or 'quit' - Exit the application")

    def _generate_quiz_question(self, context: str) -> dict:
        """Generate a quiz question from a context snippet."""
        prompt = f"""
        Based on the following context, please generate a single multiple-choice question.
        The question should test understanding of the key information in the text.

        Context:
        "{context}"

        Provide your response in a valid JSON format with the following keys:
        - "question": A string containing the question.
        - "options": A list of 4 strings representing the choices (labeled A, B, C, D).
        - "correct_answer": A string containing the letter of the correct option (e.g., "A").

        Example format:
        {{
          "question": "What is the primary function of the mitochondria?",
          "options": ["A) Protein synthesis", "B) Energy production", "C) Waste disposal", "D) Cell division"],
          "correct_answer": "B"
        }}

        Do not include any other text or explanations outside of the JSON object.
        """
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            # Clean up the response content to ensure it's a valid JSON string
            json_string = response.content.strip().replace("```json", "").replace("```", "")
            return json.loads(json_string)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"⚠️ Could not parse quiz question from LLM response: {e}")
            return None


    def _start_quiz(self):
        """Manages the quiz session with the user."""
        print("Starting quiz... Type 'stop' at any time to end the quiz.")
        status = self.vector_manager.get_knowledge_base_status()
        if not status.get('total_chunks', 0) > 0:
            print("Knowledge base is empty. Please upload documents before starting a quiz.")
            return

        while True:
            contexts = self.vector_manager.get_random_knowledge_chunks(num_chunks=1)
            if not contexts:
                print("Could not retrieve context for a question. Please ensure documents are uploaded.")
                break

            quiz_data = self._generate_quiz_question(contexts[0])
            if not quiz_data:
                print("Could not generate a question, trying again...")
                continue

            print("\n" + "="*20)
            print(quiz_data["question"])
            for option in quiz_data["options"]:
                print(f"  {option}")

            user_answer = input("Your answer (A, B, C, or D): ").strip().upper()

            if user_answer in ["STOP", "EXIT", "QUIT"]:
                break

            correct_answer = quiz_data["correct_answer"].upper()
            if user_answer == correct_answer:
                print("✅ Correct!")
            else:
                print(f"❌ Incorrect. The correct answer was {correct_answer}.")

            another = input("Another question? (y/n): ").strip().lower()
            if another != 'y':
                break

        print("Quiz finished. Returning to chat.")


    def chat(self):
        """Main chat interface."""
        print("🤖 Agentic RAG Assistant - PDF & TXT Version")
        print("📝 Created by: smgrizzlybear")
        print("📅 Date: 2025-07-12")
        print("Type 'help' for available commands or 'exit' to quit\n")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ("exit", "quit"):
                break

            # Handle special commands
            if self.handle_command(user_input):
                continue

            # Process regular chat input
            response = self.process_user_input(user_input)
            print("Agent:", response)

        print(f"\n✅ Memory and knowledge saved in {config.chroma_memory_dir} and {config.chroma_kb_dir}")

def main():
    """Entry point for the application."""
    try:
        assistant = RAGAssistant()
        assistant.chat()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()