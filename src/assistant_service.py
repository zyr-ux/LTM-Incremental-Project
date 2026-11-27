from typing import Any, Dict, List, Optional
from src.rag_assistant import RAGAssistant
from src.review_summarizer import ReviewSummarizer
from src.llm_client import LLMClient
from src.tts_voice import VoiceSynthesizer

class RetailAssistantModule:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        summarizer: Optional[ReviewSummarizer] = None,
        tts: Optional[VoiceSynthesizer] = None,
        chatbot: Optional[RAGAssistant] = None,
    ):
        self.llm_client = llm_client or LLMClient()
        self.summarizer = summarizer or ReviewSummarizer()
        self.chatbot = chatbot or RAGAssistant(llm_client=self.llm_client)
        self.tts = tts or VoiceSynthesizer()

    def summarize_reviews(self, product_name: str, reviews: List[str]) -> Dict:
        return self.summarizer.summarize_product_reviews(product_name=product_name, reviews=reviews)

    def speak_reply(self, text: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        return self.tts.speak(text=text, output_path=output_path)

    def chat(
        self,
        user_message: str,
        product_metadata: Optional[Dict] = None,
        with_voice: bool = False,
        audio_output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        rag_result = self.chatbot.ask(user_message=user_message, product_metadata=product_metadata)
        response = rag_result["answer"]
        result: Dict[str, Any] = {
            "user_message": user_message,
            "response": response,
            "sources": rag_result["sources"],
            "passages": rag_result["passages"],
            "grounded": rag_result["grounded"],
            "conversation_turns": len(self.chatbot.history) // 2,
        }
        if with_voice:
            result["audio"] = self.speak_reply(response, output_path=audio_output_path)
        return result

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

def chat_with_assistant(
    user_message: str,
    product_metadata: Optional[Dict] = None,
    with_voice: bool = False,
    audio_output_path: Optional[str] = None,
) -> Dict[str, Any]:
    assistant = get_assistant()
    return assistant.chat(
        user_message,
        product_metadata,
        with_voice=with_voice,
        audio_output_path=audio_output_path,
    )

def speak_assistant_reply(text: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    assistant = get_assistant()
    return assistant.speak_reply(text=text, output_path=output_path)
