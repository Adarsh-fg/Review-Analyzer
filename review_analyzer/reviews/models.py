from django.db import models

class Review(models.Model):
    reviewer_name = models.CharField(max_length=100)
    rating = models.IntegerField()
    review_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review by {self.reviewer_name}"