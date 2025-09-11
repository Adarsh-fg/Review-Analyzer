from transformers import pipeline
import statistics

# --- Initialize Our SINGLE, STABLE AI Model ---
# We only need the model that has proven to be 100% reliable.

print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)
print("Model loaded successfully.")


def generate_recommendation_from_complaints(complaints):
    """
    Generates a rule-based recommendation by searching for keywords in complaint texts.
    This is a stable and deterministic alternative to an unstable summarizer.
    """
    # Combine all complaint texts into one string for easy searching
    full_complaint_text = " ".join(complaints).lower()

    # Define keywords and their corresponding recommendations
    recommendation_map = {
        ("service", "support", "responsive", "refund", "runaround"): "Investigate and improve customer service response times and refund processes.",
        ("shipping", "delivery", "packaging", "arrived", "damaged"): "Review and improve the shipping and packaging process to prevent damage.",
        ("instructions", "manual", "confusing", "setup", "hard to follow"): "Rewrite or create a video guide for the product setup process to improve clarity.",
        ("battery", "charge", "drains"): "Investigate the product's battery performance and longevity.",
        ("slow", "performance", "buggy", "crashes"): "Prioritize software updates to fix bugs and improve performance.",
        ("broken", "defective", "stopped working", "abysmal"): "Improve quality control checks before the product is shipped.",
    }

    # Find the first matching keyword set and return the recommendation
    for keywords, recommendation in recommendation_map.items():
        if any(keyword in full_complaint_text for keyword in keywords):
            return recommendation
            
    # Fallback if no specific keywords are found
    return "Address the key issues raised in the detailed complaint points."


def analyze_reviews_with_ai(reviews_text):
    """
    Analyzes reviews using a definitive, stable, and AI-augmented approach:
    1. A sentiment model calculates the rating and categorizes every review.
    2. The text of the most detailed reviews are extracted directly.
    3. A rule-based system generates a stable recommendation from the extracted complaints.
    """
    if not reviews_text:
        return {
            "top_praise_points": [], "top_complaint_points": [],
            "overall_rating_suggestion": "N/A", "actionable_recommendation": "N/A"
        }

    # --- Task 1: Calculate Rating and PRE-SORT Reviews by Sentiment ---
    predicted_ratings = []
    positive_reviews = [] # For 4 and 5-star reviews
    negative_reviews = [] # For 1 and 2-star reviews

    for review in reviews_text:
        try:
            result = sentiment_pipeline(review[:512]) # Truncate for safety
            rating = int(result[0]["label"].split()[0])
            predicted_ratings.append(rating)

            if rating >= 4:
                positive_reviews.append(review)
            elif rating <= 2:
                negative_reviews.append(review)
        except Exception as e:
            print(f"Error classifying a review: {e}")
            continue

    overall_rating_suggestion = (
        f"Approximately {statistics.mean(predicted_ratings):.2f} out of 5 stars, based on sentiment analysis."
        if predicted_ratings
        else "Could not determine rating from reviews."
    )

    # --- Task 2: EXTRACT Praise and Complaint points directly ---
    positive_reviews.sort(key=len, reverse=True)
    negative_reviews.sort(key=len, reverse=True)
    top_praise_points = positive_reviews[:3] or ["No praise points found in 4-5 star reviews."]
    top_complaint_points = negative_reviews[:3] or ["No complaint points found in 1-2 star reviews."]
    
    # --- Task 3: Generate Recommendation with the STABLE, Rule-Based System ---
    actionable_recommendation = generate_recommendation_from_complaints(top_complaint_points)

    return {
        "top_praise_points": top_praise_points,
        "top_complaint_points": top_complaint_points,
        "overall_rating_suggestion": overall_rating_suggestion,
        "actionable_recommendation": actionable_recommendation,
    }