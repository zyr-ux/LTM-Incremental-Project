from typing import Dict, List, Optional, Union
from src.support_chatbot import SupportChatbot
from src.review_summarizer import ReviewSummarizer
from src.llm_client import LLMClient

class RetailAssistantModule:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        summarizer: Optional[ReviewSummarizer] = None
    ):
        self.llm_client = llm_client or LLMClient()
        self.summarizer = summarizer or ReviewSummarizer()
        self.chatbot = SupportChatbot(llm_client=self.llm_client)

    def summarize_reviews(self, product_name: str, reviews: List[str]) -> Dict:
        return self.summarizer.summarize_product_reviews(product_name=product_name, reviews=reviews)

    def chat(self, user_message: str, product_metadata: Optional[Dict] = None) -> Dict:
        response = self.chatbot.reply(user_message=user_message, product_metadata=product_metadata)
        return {
            "user_message": user_message,
            "response": response,
            "conversation_turns": len(self.chatbot.history) // 2
        }

    def reset_chat(self):
        self.chatbot.clear_history()

_default_assistant: Optional[RetailAssistantModule] = None

def get_assistant() -> RetailAssistantModule:
    global _default_assistant
    if _default_assistant is None:
        _default_assistant = RetailAssistantModule()
    return _default_assistant

def summarize_product_reviews(product_name: str, reviews: List[str]) -> Dict:
    assistant = get_assistant()
    return assistant.summarize_reviews(product_name, reviews)

def chat_with_assistant(user_message: str, product_metadata: Optional[Dict] = None) -> Dict:
    assistant = get_assistant()
    return assistant.chat(user_message, product_metadata)
