import os
import statistics
import google.generativeai as genai
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

# Model 1 (for Ratings and Categorization):
print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Model 2 (for Recommendation): The powerful Gemini model.
print("Configuring Gemini model...")
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Gemini model configured successfully.")
except Exception as e:
    print(f"CRITICAL ERROR configuring Gemini: {e}")
    gemini_model = None


def generate_recommendation_with_gemini(praise, complaints):
    """
    Generates an actionable recommendation using the Google Gemini API,
    dynamically adjusting its prompt based on the sentiment of the feedback.
    """
    if not gemini_model:
        return "Gemini model is not available. Please check API key configuration."

    NO_PRAISE_MSG = "No praise points found in 4-5 star reviews."
    NO_COMPLAINTS_MSG = "No complaint points found in 1-2 star reviews."

    has_praise = praise and praise[0] != NO_PRAISE_MSG
    has_complaints = complaints and complaints[0] != NO_COMPLAINTS_MSG

    if not has_praise and not has_complaints:
        return "Not enough specific feedback was provided to generate a recommendation."

    prompt_parts = ["Analyze the following customer feedback and provide a concise, actionable business recommendation."]

    if has_praise:
        praise_text = "\n- ".join(praise)
        prompt_parts.append(f"\nPositive Feedback (Strengths):\n- {praise_text}")

    if has_complaints:
        complaint_text = "\n- ".join(complaints)
        prompt_parts.append(f"\nNegative Feedback (Areas for Improvement):\n- {complaint_text}")

    if has_praise and not has_complaints:
        prompt_parts.append("\nThe feedback is overwhelmingly positive. Provide an encouraging recommendation on how the business can continue to build on this success.")
    elif not has_praise and has_complaints:
        prompt_parts.append("\nThe feedback is negative. Focus the recommendation on addressing these critical issues.")
    else: 
        prompt_parts.append("\nThe feedback is mixed. Provide a balanced recommendation that leverages strengths to address weaknesses.")

    prompt_parts.append("\nActionable Recommendation:")
    prompt = "\n".join(prompt_parts)

    try:
        response = gemini_model.generate_content(prompt)
        
        if not response.parts:
            if response.prompt_feedback.block_reason:
                print(f"Gemini API Blocked Prompt. Reason: {response.prompt_feedback.block_reason}")
                return f"Recommendation could not be generated due to a content safety block ({response.prompt_feedback.block_reason})."
            else:
                return "Recommendation could not be generated (empty response)."

        return response.text.strip()
    except Exception as e:
        print(f"\n--- GEMINI API ERROR: {e} ---\n")
        return "Could not generate AI recommendation due to an API error. (Check server logs for details)."


def analyze_reviews_with_ai(reviews_text):
    """
    Analyzes a list of review texts, categorizes them, and generates
    a comprehensive analysis including an AI-powered recommendation.
    """
    if not reviews_text:
        return {
            "top_praise_points": [], "top_complaint_points": [],
            "overall_rating_suggestion": "N/A", "actionable_recommendation": "N/A"
        }

    predicted_ratings, positive_reviews, negative_reviews = [], [], []
    for review in reviews_text:
        try:
            result = sentiment_pipeline(review[:512])
            rating = int(result[0]["label"].split()[0])
            predicted_ratings.append(rating)
            if rating >= 4:
                positive_reviews.append(review)
            else:
                negative_reviews.append(review)
        except Exception as e:
            print(f"Error classifying a review: {e}")
            continue

    overall_rating_suggestion = (
        f"Approximately {statistics.mean(predicted_ratings):.2f} out of 5 stars."
        if predicted_ratings else "Could not determine rating."
    )

    positive_reviews.sort(key=len, reverse=True)
    negative_reviews.sort(key=len, reverse=True)

    top_praise_points = positive_reviews[:3] or ["No praise points found in 4-5 star reviews."]
    top_complaint_points = negative_reviews[:3] or ["No complaint points found in 1-2 star reviews."]
    
    actionable_recommendation = generate_recommendation_with_gemini(top_praise_points, top_complaint_points)

    return {
        "top_praise_points": top_praise_points,
        "top_complaint_points": top_complaint_points,
        "overall_rating_suggestion": overall_rating_suggestion,
        "actionable_recommendation": actionable_recommendation,
    }