from src.assistant_service import RetailAssistantModule
from src.rag_assistant import NO_CONTEXT_ANSWER, RAGAssistant


class FakeKnowledgeBase:
    def __init__(self, results):
        self.results = results

    def search(self, query: str, top_k: int = 4):
        return self.results[:top_k]


class FakeLLMClient:
    def generate(self, prompt: str, system_prompt: str = "", max_tokens: int = 250) -> str:
        if "30 days" in prompt:
            return "You can return unworn items within 30 days [returns_policy.md]."
        return "Express shipping takes 1-2 business days [shipping_policy.md]."


def test_rag_assistant_returns_grounded_answer_and_passages():
    passages = [
        {
            "score": 0.85,
            "text": "Customers can return unworn items with tags within 30 days of delivery.",
            "metadata": {"source": "returns_policy.md", "source_type": "policy"},
        }
    ]
    assistant = RAGAssistant(
        knowledge_base=FakeKnowledgeBase(passages),
        llm_client=FakeLLMClient(),
    )

    result = assistant.ask("What is your return policy?")

    assert result["grounded"] is True
    assert "30 days" in result["answer"]
    assert result["sources"] == ["returns_policy.md"]
    assert len(result["passages"]) == 1
    assert len(assistant.get_history()) == 2


def test_rag_assistant_avoids_hallucination_when_context_missing():
    assistant = RAGAssistant(
        knowledge_base=FakeKnowledgeBase([]),
        llm_client=FakeLLMClient(),
    )

    result = assistant.ask("Do you sell Mars rover parts?")

    assert result["grounded"] is False
    assert result["answer"] == NO_CONTEXT_ANSWER
    assert result["sources"] == []
    assert result["passages"] == []


def test_rag_assistant_filters_low_score_passages():
    low_score_passages = [
        {
            "score": 0.05,
            "text": "Unrelated text",
            "metadata": {"source": "other.md"},
        }
    ]
    assistant = RAGAssistant(
        knowledge_base=FakeKnowledgeBase(low_score_passages),
        llm_client=FakeLLMClient(),
        min_score=0.2,
    )

    result = assistant.ask("What is the warranty on laptops?")

    assert result["grounded"] is False
    assert result["passages"] == []


def test_assistant_module_uses_rag_assistant():
    passages = [
        {
            "score": 0.91,
            "text": "Express shipping takes 1-2 business days for $12.99.",
            "metadata": {"source": "shipping_policy.md", "source_type": "policy"},
        }
    ]
    rag = RAGAssistant(
        knowledge_base=FakeKnowledgeBase(passages),
        llm_client=FakeLLMClient(),
    )
    module = RetailAssistantModule(chatbot=rag)

    chat_result = module.chat("How fast is express shipping?")

    assert chat_result["grounded"] is True
    assert chat_result["sources"] == ["shipping_policy.md"]
    assert len(chat_result["passages"]) == 1
    assert chat_result["conversation_turns"] == 1

    module.reset_chat()
    assert len(rag.get_history()) == 0
