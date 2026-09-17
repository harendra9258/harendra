from transformers import pipeline

generator= pipeline(
    "text-generation",
    model="distilgpt2"
)

result= generator("Artificial Intelligence is", max_new_tokens=50)

print(result)