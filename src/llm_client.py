import os
from typing import Dict, List, Optional
import httpx

class LLMClient:
    def __init__(self, provider: str = "openai", api_key: Optional[str] = None):
        self.provider = provider.lower()
        if self.provider == "anthropic":
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        else:
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def generate(self, prompt: str, system_prompt: Optional[str] = None, max_tokens: int = 250) -> str:
        if not self.is_configured():
            return self._fallback_rule_generation(prompt, system_prompt)
        try:
            if self.provider == "anthropic":
                return self._call_anthropic(prompt, system_prompt, max_tokens)
            return self._call_openai(prompt, system_prompt, max_tokens)
        except Exception:
            return self._fallback_rule_generation(prompt, system_prompt)

    def _call_openai(self, prompt: str, system_prompt: Optional[str], max_tokens: int) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        messages = []
        if bool(system_prompt):
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.2
        }
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_content = data["choices"][0]["message"]["content"]
            content: str = str(raw_content)
            return content.strip()

    def _call_anthropic(self, prompt: str, system_prompt: Optional[str], max_tokens: int) -> str:
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307"),
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2
        }
        if bool(system_prompt):
            payload["system"] = system_prompt
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            raw_content = data["content"][0]["text"]
            content: str = str(raw_content)
            return content.strip()

    def _fallback_rule_generation(self, prompt: str, system_prompt: Optional[str]) -> str:
        lowered = prompt.lower()
        if "return" in lowered or "refund" in lowered:
            return "RetailMind Support: Our policy allows hassle-free returns within 30 days of delivery. Once received, refunds are processed within 3-5 business days."
        if "shipping" in lowered or "delivery" in lowered:
            return "RetailMind Support: Standard domestic delivery takes 3-5 business days. Express shipping delivers within 1-2 business days with live tracking."
        if "size" in lowered or "fit" in lowered or "shirt" in lowered or "pants" in lowered or "jeans" in lowered:
            return "RetailMind Support: Our garments fit true to standard size specifications. Please consult the product size chart or order one size up if you prefer a relaxed fit."
        if "defect" in lowered or "damaged" in lowered or "broken" in lowered:
            return "RetailMind Support: We sincerely apologize for the quality issue. We offer immediate replacements or refunds for damaged goods within 48 hours of reporting."
        return "RetailMind Support: Thank you for reaching out! Our customer team is here to assist with catalog details, orders, and store policies."
