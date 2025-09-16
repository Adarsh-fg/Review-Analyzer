import os
import re 
import statistics
import google.generativeai as genai
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

# --- Global Variables & Model Loading ---

# Keywords for the new helper functions
NEGATIVE_KEYWORDS = [
    "broken", "damaged", "defective", "scratched", "wrong size",
    "late", "delay", "missing", "fake", "cheap", "scam", "fraud",
    "expensive", "not worth", "refund", "waste of money"
]

# Model 1 (for Ratings and Categorization):
print("Loading sentiment analysis model...")
sentiment_pipeline = pipeline(
    "text-classification",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# NEW: Model 2 (for Summarization): Using your specified BART model
print("Loading summarization model (BART)...")
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# Model 3 (for Recommendation): The powerful Gemini model.
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

# YOUR PROVIDED HELPER & SUMMARIZATION FUNCTIONS

def filter_positive_sentences(feedback_list):
    positive_sentences = []
    for text in feedback_list:
        sentences = re.split(r'[.!?]', text)
        for s in sentences:
            s_clean = s.strip()
            if s_clean and not any(kw in s_clean.lower() for kw in NEGATIVE_KEYWORDS):
                positive_sentences.append(s_clean)
    return positive_sentences

def filter_negative_sentences(feedback_list):
    negative_sentences = []
    for text in feedback_list:
        sentences = re.split(r'[.!?]', text)
        for s in sentences:
            s_clean = s.strip()
            if s_clean and any(kw in s_clean.lower() for kw in NEGATIVE_KEYWORDS):
                negative_sentences.append(s_clean)
    return negative_sentences

def summarize_praises(feedback_list):
    positive_sentences = filter_positive_sentences(feedback_list)
    if not positive_sentences:
        return "None"

    text_to_summarize = ". ".join(positive_sentences)
    try:
        result = summarizer(text_to_summarize, max_length=130, min_length=25, do_sample=False)
        return result[0]["summary_text"].strip()
    except Exception as e:
        print(f"Summarizer error (praises): {e}")
        return "Could not generate summary."

def summarize_pain_points(feedback_list):
    negative_sentences = filter_negative_sentences(feedback_list)
    if not negative_sentences:
        return "None"

    text_to_summarize = ". ".join(negative_sentences)
    try:
        result = summarizer(text_to_summarize, max_length=130, min_length=25, do_sample=False)
        return result[0]["summary_text"].strip()
    except Exception as e:
        print(f"Summarizer error (pain points): {e}")
        return "Could not generate summary."


# GEMINI RECOMMENDATION FUNCTION 

def generate_recommendation_with_gemini(praise, complaints):
    if not gemini_model:
        return ["Gemini model is not available. Please check API key configuration."]

    NO_PRAISE_MSG = "No specific praise points found."
    NO_COMPLAINTS_MSG = "No specific complaint points found."

    has_praise = praise and praise[0] != NO_PRAISE_MSG
    has_complaints = complaints and complaints[0] != NO_COMPLAINTS_MSG

    if not has_praise and not has_complaints:
        return ["Not enough specific feedback was provided to generate a recommendation."]

    prompt_parts = ["Analyze the following summarized customer feedback and provide a simple to-do list with short, direct action points using easy-to-understand words."]

    if has_praise:
        praise_summary = "\n".join(praise)
        prompt_parts.append(f"\nPositive Summary:\n{praise_summary}")

    if has_complaints:
        complaint_summary = "\n".join(complaints)
        prompt_parts.append(f"\nNegative Summary:\n{complaint_summary}")

    prompt_parts.append("\nActionable To-Do List (Start each point with '*'):")
    prompt = "\n".join(prompt_parts)

    try:
        response = gemini_model.generate_content(prompt)
        if not response.parts:
            return [f"Recommendation could not be generated due to a content safety block ({response.prompt_feedback.block_reason})."] if response.prompt_feedback.block_reason else ["Recommendation could not be generated."]
        
        recommendation_text = response.text.strip()
        bullet_points = [line.strip() for line in recommendation_text.split('\n') if line.strip().startswith(('*', '-'))]
        cleaned_points = [point.lstrip('*- ').strip() for point in bullet_points]
        return cleaned_points if cleaned_points else [recommendation_text]
    except Exception as e:
        print(f"\n--- GEMINI API ERROR: {e} ---\n")
        return ["Could not generate AI recommendation due to an API error."]


# MAIN ANALYSIS FUNCTION (MODIFIED)

def analyze_reviews_with_ai(reviews_text):
    """
    Analyzes reviews, but now uses the keyword-based summarizer
    to generate the praise and complaint points.
    """
    if not reviews_text:
        return {"top_praise_points": [], "top_complaint_points": [], "overall_rating_suggestion": "N/A", "actionable_recommendation": "N/A"}

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

    overall_rating_suggestion = (f"Approximately {statistics.mean(predicted_ratings):.2f} out of 5 stars." if predicted_ratings else "Could not determine rating.")

    summarized_praise = summarize_praises(positive_reviews)
    summarized_complaints = summarize_pain_points(negative_reviews)

    top_praise_points = [summarized_praise] if summarized_praise != "None" else ["No specific praise points found."]
    top_complaint_points = [summarized_complaints] if summarized_complaints != "None" else ["No specific complaint points found."]
    
    actionable_recommendation = generate_recommendation_with_gemini(top_praise_points, top_complaint_points)

    return {
        "top_praise_points": top_praise_points,
        "top_complaint_points": top_complaint_points,
        "overall_rating_suggestion": overall_rating_suggestion,
        "actionable_recommendation": actionable_recommendation,
    }