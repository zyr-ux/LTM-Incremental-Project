from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.knowledge_base import RetailKnowledgeBase
from src.llm_client import LLMClient

RAG_SYSTEM_PROMPT = """You are RetailMind Assistant, a helpful retail support and staff Q&A assistant.
Answer the question using ONLY the retrieved context below.
Cite the source name in brackets, like [returns_policy.md], for every fact you use.
If the context does not contain enough information to answer, say:
"I don't have enough information in the RetailMind knowledge base to answer that. Please check with human support at support@retailmind.ai."
Do not guess or make up policies, prices, or product details."""

RAG_USER_TEMPLATE = """Retrieved Context:
{context}

{product_note}Question: {question}

Grounded Answer:"""

NO_CONTEXT_ANSWER = (
    "I don't have enough information in the RetailMind knowledge base to answer that. "
    "Please check with human support at support@retailmind.ai."
)


def format_passages(passages: List[Dict[str, Any]]) -> str:
    lines = []
    for index, passage in enumerate(passages, start=1):
        source = passage.get("metadata", {}).get("source", "unknown")
        score = passage.get("score", 0.0)
        text = passage.get("text", "")
        lines.append(f"[{index}] Source: {source} (score: {score:.3f})\n{text}")
    return "\n\n".join(lines)


def format_product_note(product_metadata: Optional[Dict[str, Any]]) -> str:
    if not product_metadata:
        return ""
    name = product_metadata.get("productDisplayName", "Unknown")
    category = product_metadata.get("masterCategory", "")
    color = product_metadata.get("baseColour", "N/A")
    return f"Active Product: {name} | Category: {category} | Color: {color}\n\n"


class RAGAssistant:
    def __init__(
        self,
        knowledge_base: Optional[RetailKnowledgeBase] = None,
        llm_client: Optional[LLMClient] = None,
        top_k: int = 4,
        min_score: float = 0.2,
    ):
        self.knowledge_base = knowledge_base or RetailKnowledgeBase()
        self.llm_client = llm_client or LLMClient()
        self.top_k = top_k
        self.min_score = min_score
        self.history: List[Dict[str, str]] = []
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", RAG_USER_TEMPLATE),
        ])
        self.chain = (
            self.prompt
            | RunnableLambda(self._call_llm)
            | StrOutputParser()
        )

    def _call_llm(self, prompt_value: Any) -> str:
        messages = prompt_value.to_messages()
        system_prompt = messages[0].content if len(messages) > 0 else ""
        user_prompt = messages[1].content if len(messages) > 1 else str(prompt_value)
        return self.llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)

    def clear_history(self) -> None:
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.history)

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        try:
            results = self.knowledge_base.search(question, top_k=self.top_k)
        except FileNotFoundError:
            return []
        return [item for item in results if item.get("score", 0.0) >= self.min_score]

    def ask(
        self,
        user_message: str,
        product_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        clean_question = user_message.strip()
        if not clean_question:
            return {
                "answer": "How can I help you today with your retail purchase or order?",
                "sources": [],
                "passages": [],
                "grounded": False,
            }

        passages = self.retrieve(clean_question)
        if not passages:
            self._record_turn(clean_question, NO_CONTEXT_ANSWER)
            return {
                "answer": NO_CONTEXT_ANSWER,
                "sources": [],
                "passages": [],
                "grounded": False,
            }

        context_text = format_passages(passages)
        product_note = format_product_note(product_metadata)
        answer = self.chain.invoke({
            "context": context_text,
            "product_note": product_note,
            "question": clean_question,
        })

        sources = []
        for passage in passages:
            source = passage.get("metadata", {}).get("source", "unknown")
            if source not in sources:
                sources.append(source)

        self._record_turn(clean_question, answer)
        return {
            "answer": answer,
            "sources": sources,
            "passages": passages,
            "grounded": True,
        }

    def reply(
        self,
        user_message: str,
        product_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        return self.ask(user_message=user_message, product_metadata=product_metadata)["answer"]

    def _record_turn(self, user_message: str, assistant_answer: str) -> None:
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_answer})
