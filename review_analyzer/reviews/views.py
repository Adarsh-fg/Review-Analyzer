from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
from .models import Review
from .serializers import ReviewSerializer
from .analysis import analyze_reviews_with_ai

def index(request):
    return render(request, 'index.html')

@api_view(['GET', 'POST'])
def review_list(request):
    if request.method == 'GET':
        reviews = Review.objects.all()
        serializer = ReviewSerializer(reviews, many=True)
        return Response(serializer.data)
    elif request.method == 'POST':
        serializer = ReviewSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=201)
        return Response(serializer.errors, status=400)
    
@api_view(['POST'])
def analyze_reviews(request):
    # Fetch all reviews from the database
    reviews = Review.objects.all()

    # If there are no reviews, return an empty analysis
    if not reviews.exists():
        return Response({
            "top_praise_points": [],
            "top_complaint_points": [],
            "overall_rating_suggestion": "No reviews to analyze.",
            "actionable_recommendation": "Submit some reviews first."
        })

    # Extract the text from each review
    review_texts = [review.review_text for review in reviews]

    # Call our AI analysis function
    analysis_result = analyze_reviews_with_ai(review_texts)

    return Response(analysis_result)