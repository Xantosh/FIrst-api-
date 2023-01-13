from rest_framework import serializers

class BookSerializer(serializers.Serializer):
    id = models.IntegerField(primary_key=True)
    title = models.CharField(max_length=200)
    author = models.CharField(max_length=200)
