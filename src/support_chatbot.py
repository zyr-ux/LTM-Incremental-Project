from typing import Dict, List, Optional, Tuple
from src.llm_client import LLMClient

DEFAULT_POLICY_TEXT = """
RetailMind Store Policies:
1. Returns & Exchanges: Customers can return or exchange any unworn, unwashed item with tags attached within 30 days of delivery.
2. Refunds: Approved returns will be refunded to the original payment method within 3 to 5 business days after inspection.
3. Shipping: Standard domestic shipping takes 3-5 business days ($4.99 or free on orders over $50). Express shipping takes 1-2 business days ($12.99).
4. Order Tracking: Tracking numbers are emailed immediately once the order ships from our fulfillment center.
5. Damaged or Defective Items: If an item arrives damaged, report it within 48 hours for an instant replacement or full refund.
"""

SYSTEM_PROMPT_TEMPLATE = """You are RetailMind Assistant, an intelligent, helpful, and courteous AI shopping and customer support assistant for RetailMind.
Always stay on-brand, friendly, and concise.
Ground your answers strictly on the product catalog information and store policies provided below.
If you do not know the answer based on the provided context, politely advise the customer to reach human customer support at support@retailmind.ai.

Store Policies:
{policy_text}

Active Product Details:
{product_context}
"""

class SupportChatbot:
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        policy_text: str = DEFAULT_POLICY_TEXT
    ):
        self.llm_client = llm_client or LLMClient()
        self.policy_text = policy_text.strip()
        self.history: List[Dict[str, str]] = []

    def clear_history(self):
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.history)

    def _build_context_prompt(self, user_message: str, product_metadata: Optional[Dict] = None) -> Tuple[str, str]:
        product_context = "No specific product selected."
        if bool(product_metadata):
            product_context = (
                f"Product Name: {product_metadata.get('productDisplayName', 'Unknown')}\n"
                f"Category: {product_metadata.get('masterCategory', '')} - {product_metadata.get('subCategory', '')}\n"
                f"Color: {product_metadata.get('baseColour', 'N/A')}\n"
                f"Usage/Season: {product_metadata.get('usage', '')} / {product_metadata.get('season', '')}"
            )
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            policy_text=self.policy_text,
            product_context=product_context
        )
        history_lines = []
        for msg in self.history[-6:]:
            role = "Customer" if msg["role"] == "user" else "RetailMind Assistant"
            history_lines.append(f"{role}: {msg['content']}")
        history_lines.append(f"Customer: {user_message}")
        user_prompt = "\n".join(history_lines) + "\nRetailMind Assistant:"
        return system_prompt, user_prompt

    def reply(self, user_message: str, product_metadata: Optional[Dict] = None) -> str:
        clean_input = user_message.strip()
        if not clean_input:
            return "How can I help you today with your retail purchase or order?"
        system_prompt, user_prompt = self._build_context_prompt(clean_input, product_metadata)
        response_text = self.llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)
        self.history.append({"role": "user", "content": clean_input})
        self.history.append({"role": "assistant", "content": response_text})
        return response_text
