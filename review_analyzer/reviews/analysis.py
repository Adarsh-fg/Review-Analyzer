import os
import statistics
import google.generativeai as genai
from transformers import pipeline
from dotenv import load_dotenv

# --- Load Environment Variables (for the API Key) ---
load_dotenv()

# --- Initialize Our TWO Specialist AI Models ---

# Model 1 (for Ratings and Categorization): 100% reliable.
print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Model 2 (for Recommendation): The powerful Gemini Pro model.
print("Configuring Gemini Pro model...")
try:
    # It's a good practice to check if the key exists before configuring
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file.")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("Gemini Pro model configured successfully.")
except Exception as e:
    print(f"CRITICAL ERROR configuring Gemini: {e}")
    gemini_model = None


def generate_recommendation_with_gemini(complaints):
    """
    Generates a single, actionable recommendation using the Google Gemini API,
    with robust error logging to diagnose failures.
    """
    if not gemini_model:
        return "Gemini model is not available. Please check API key configuration and server logs."
        
    if not complaints or "No complaint points" in complaints[0]:
        return "No specific complaints were identified to generate a recommendation from."

    full_complaint_text = "\n- ".join(complaints)

    # A slightly safer, more analytical prompt to avoid content filters.
    prompt = f"""
Analyze the following customer complaints and provide a single, actionable business recommendation.

Complaints:
- {full_complaint_text}

Recommendation:
"""

    try:
        response = gemini_model.generate_content(prompt)
        # It's possible the response is generated but is empty or blocked.
        # We check the prompt_feedback for safety issues.
        if not response.parts:
            if response.prompt_feedback.block_reason:
                print(f"Gemini API Blocked Prompt. Reason: {response.prompt_feedback.block_reason}")
                return f"Recommendation could not be generated due to a content safety block ({response.prompt_feedback.block_reason})."
            else:
                return "Recommendation could not be generated (empty response)."

        return response.text.strip()
    except Exception as e:
        # --- THIS IS THE MOST IMPORTANT CHANGE ---
        # We will now print the EXACT error message from the Gemini API.
        print("\n--- GEMINI API ERROR ---")
        print(f"The call to the Gemini API failed with a specific error: {e}")
        print("--- END GEMINI API ERROR ---\n")
        return "Could not generate AI recommendation due to an API error. (Check server logs for details)."


def analyze_reviews_with_ai(reviews_text):
    """
    The main analysis function. (No changes needed here).
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
            elif rating <= 2:
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
    
    actionable_recommendation = generate_recommendation_with_gemini(top_complaint_points)

    return {
        "top_praise_points": top_praise_points,
        "top_complaint_points": top_complaint_points,
        "overall_rating_suggestion": overall_rating_suggestion,
        "actionable_recommendation": actionable_recommendation,
    }