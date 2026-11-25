from src.llm_client import LLMClient
from src.review_summarizer import ReviewSummarizer
from src.support_chatbot import SupportChatbot
from src.assistant_service import RetailAssistantModule, chat_with_assistant, summarize_product_reviews

def test_llm_client_fallback():
    client = LLMClient(api_key="")
    assert not client.is_configured()
    response = client.generate("What is your refund policy?")
    assert "refund" in response.lower() or "30 days" in response.lower()

def test_support_chatbot_conversation_history():
    bot = SupportChatbot()
    assert len(bot.get_history()) == 0
    resp1 = bot.reply("What is your return policy?")
    assert len(bot.get_history()) == 2
    assert bot.get_history()[0]["role"] == "user"
    assert bot.get_history()[1]["role"] == "assistant"
    
    product = {
        "productDisplayName": "Peter England Men Party Blue Jeans",
        "masterCategory": "Apparel",
        "subCategory": "Bottomwear",
        "baseColour": "Blue"
    }
    resp2 = bot.reply("How does this shirt or jeans fit?", product_metadata=product)
    assert len(bot.get_history()) == 4
    bot.clear_history()
    assert len(bot.get_history()) == 0

def test_review_summarizer():
    summarizer = ReviewSummarizer()
    reviews = [
        "Great quality shirt, fit is perfect and the material is soft.",
        "Loved the color and comfortable fabric. Fast shipping too!",
        "Excellent price and durable build, highly recommend."
    ]
    result = summarizer.summarize_product_reviews("Turtle Check Men Navy Blue Shirt", reviews)
    assert result["product_name"] == "Turtle Check Men Navy Blue Shirt"
    assert result["total_reviews"] == 3
    assert len(result["theme_summary"]) > 0
    assert "strengths" in result["detected_themes"]

def test_assistant_module_facade():
    assistant = RetailAssistantModule()
    chat_res = assistant.chat("Do you offer express shipping?")
    assert "response" in chat_res
    assert chat_res["conversation_turns"] == 1
    
    reviews = ["Terrible material, broke after one day.", "Very poor fit, totally disappointed."]
    summary_res = assistant.summarize_reviews("Sample Product", reviews)
    assert summary_res["total_reviews"] == 2
    assert "concerns" in summary_res["detected_themes"]
