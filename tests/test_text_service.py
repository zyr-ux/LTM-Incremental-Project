from src.text_service import analyze_review_sentiment, route_support_ticket


def test_sentiment_positive_review():
    review = "Outstanding product, super fast delivery and works flawlessly!"
    result: dict = analyze_review_sentiment(review)  # type: ignore
    assert result["sentiment"] == "Positive"
    assert result["confidence"] > 0.8
    assert "Negative" in result["probabilities"]
    assert "Positive" in result["probabilities"]


def test_sentiment_negative_review():
    review = "Completely defective and arrived with broken parts. Absolute waste of money."
    result: dict = analyze_review_sentiment(review)  # type: ignore
    assert result["sentiment"] == "Negative"
    assert result["confidence"] > 0.8


def test_sentiment_batch_input():
    reviews = [
        "Loved it! High quality finish and great packaging.",
        "Terrible and damaged on arrival, unusable.",
    ]
    results: list = analyze_review_sentiment(reviews)  # type: ignore
    assert len(results) == 2
    assert results[0]["sentiment"] == "Positive"
    assert results[1]["sentiment"] == "Negative"


def test_route_delivery_ticket():
    ticket = "Where is my shipment? Tracking number RM-832190 has had no updates for five days."
    result: dict = route_support_ticket(ticket)  # type: ignore
    assert result["routed_department"] == "Delivery & Tracking"
    assert result["priority"] == "Standard"
    assert result["confidence"] > 0.7


def test_route_billing_ticket():
    ticket = "I was charged twice on my credit card for invoice RM-449102. Please issue a refund."
    result: dict = route_support_ticket(ticket)  # type: ignore
    assert result["routed_department"] == "Refund & Billing"
    assert result["priority"] == "High"
    assert result["confidence"] > 0.7


def test_route_product_defect_ticket():
    ticket = "The kitchen blender motor burned out and stopped working after two uses."
    result: dict = route_support_ticket(ticket)  # type: ignore
    assert result["routed_department"] == "Product Quality & Defect"
    assert result["priority"] == "Standard"


def test_route_account_security_ticket():
    ticket = "Suspicious login alert received. My account password was changed from an unrecognized device."
    result: dict = route_support_ticket(ticket)  # type: ignore
    assert result["routed_department"] == "Account & Security"
    assert result["priority"] == "High"
