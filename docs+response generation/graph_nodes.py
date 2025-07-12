from typing import TypedDict, Optional
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import HumanMessage
from vector_store_manager import VectorStoreManager

# State schema
class AgentState(TypedDict):
    input: str
    memory: Optional[str]
    knowledge: Optional[str]
    response: Optional[str]

class GraphNodes:
    """Contains all the node functions for the LangGraph workflow."""
    
    def __init__(self, llm: ChatMistralAI, vector_manager: VectorStoreManager):
        self.llm = llm
        self.vector_manager = vector_manager
    
    def retrieve_memory_node(self, state: AgentState) -> AgentState:
        """Node to retrieve relevant memory."""
        query = state["input"]
        memory = self.vector_manager.load_memory_variables(query)
        return {**state, "memory": memory}
    
    def retrieve_knowledge_node(self, state: AgentState) -> AgentState:
        """Node to retrieve relevant knowledge."""
        query = state["input"]
        knowledge = self.vector_manager.retrieve_knowledge(query)
        return {**state, "knowledge": knowledge}
    
    def generate_response_node(self, state: AgentState) -> AgentState:
        """Node to generate LLM response."""
        memory = state.get("memory", "")
        knowledge = state.get("knowledge", "")
        query = state["input"]

        prompt = "\n\n".join(filter(None, [
            f"Knowledge:\n{knowledge}" if knowledge else "",
            f"Conversation History:\n{memory}" if memory else "",
            f"User: {query}"
        ]))

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {**state, "response": response.content}
    
    def update_memory_node(self, state: AgentState) -> AgentState:
        """Node to update memory with conversation."""
        self.vector_manager.save_memory_context(
            state["input"], 
            state["response"]
        )
        return {}