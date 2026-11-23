import unittest
from src.text_service import analyze_review_sentiment, route_support_ticket

class TestTextService(unittest.TestCase):
    def test_sentiment_positive_review(self):
        review = "Outstanding product, super fast delivery and works flawlessly!"
        result: dict = analyze_review_sentiment(review)  # type: ignore
        self.assertEqual(result["sentiment"], "Positive")
        self.assertGreater(result["confidence"], 0.8)
        self.assertIn("Negative", result["probabilities"])
        self.assertIn("Positive", result["probabilities"])

    def test_sentiment_negative_review(self):
        review = "Completely defective and arrived with broken parts. Absolute waste of money."
        result: dict = analyze_review_sentiment(review)  # type: ignore
        self.assertEqual(result["sentiment"], "Negative")
        self.assertGreater(result["confidence"], 0.8)

    def test_sentiment_batch_input(self):
        reviews = [
            "Loved it! High quality finish and great packaging.",
            "Terrible and damaged on arrival, unusable."
        ]
        results: list = analyze_review_sentiment(reviews)  # type: ignore
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["sentiment"], "Positive")
        self.assertEqual(results[1]["sentiment"], "Negative")

    def test_route_delivery_ticket(self):
        ticket = "Where is my shipment? Tracking number RM-832190 has had no updates for five days."
        result: dict = route_support_ticket(ticket)  # type: ignore
        self.assertEqual(result["routed_department"], "Delivery & Tracking")
        self.assertEqual(result["priority"], "Standard")
        self.assertGreater(result["confidence"], 0.7)

    def test_route_billing_ticket(self):
        ticket = "I was charged twice on my credit card for invoice RM-449102. Please issue a refund."
        result: dict = route_support_ticket(ticket)  # type: ignore
        self.assertEqual(result["routed_department"], "Refund & Billing")
        self.assertEqual(result["priority"], "High")
        self.assertGreater(result["confidence"], 0.7)

    def test_route_product_defect_ticket(self):
        ticket = "The kitchen blender motor burned out and stopped working after two uses."
        result: dict = route_support_ticket(ticket)  # type: ignore
        self.assertEqual(result["routed_department"], "Product Quality & Defect")
        self.assertEqual(result["priority"], "Standard")

    def test_route_account_security_ticket(self):
        ticket = "Suspicious login alert received. My account password was changed from an unrecognized device."
        result: dict = route_support_ticket(ticket)  # type: ignore
        self.assertEqual(result["routed_department"], "Account & Security")
        self.assertEqual(result["priority"], "High")

if __name__ == "__main__":
    unittest.main()
