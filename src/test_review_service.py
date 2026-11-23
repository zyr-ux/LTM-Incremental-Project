from review_service import ReviewSentimentService

def test_sentiment_service():
    service = ReviewSentimentService()

    pos_sample = "This product exceeded my expectations! Super easy to use and great quality."
    pos_result = service.analyze_review(pos_sample)
    print("Positive sample result:", pos_result)
    assert pos_result["sentiment"] == "Positive"
    assert pos_result["confidence"] > 0.5

    neg_sample = "Terrible experience. The item arrived broken, useless waste of money."
    neg_result = service.analyze_review(neg_sample)
    print("Negative sample result:", neg_result)
    assert neg_result["sentiment"] == "Negative"
    assert neg_result["confidence"] > 0.5

    print("All review sentiment tests passed successfully!")

if __name__ == "__main__":
    test_sentiment_service()
