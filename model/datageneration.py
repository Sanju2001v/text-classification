import pandas as pd
import random

# Sample text data
sample_texts = [
    "I love this product, it's amazing!",
    "Terrible service, will never use again.",
    "The food was fantastic, highly recommend it.",
    "The delivery was late and the package was damaged.",
    "Great experience, very happy with my purchase.",
    "Awful quality, not worth the money.",
    "The app is easy to use and very intuitive.",
    "The customer support is unresponsive and rude.",
    "Excellent quality, exceeded my expectations.",
    "Disappointing experience, won't buy again."
]

# Generate 100 data points
data = {
    "text": [random.choice(sample_texts) for _ in range(100)],
    "label": [random.choice([0, 1]) for _ in range(100)]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("dataset.csv", index=False)

print("Sample dataset created and saved as 'dataset.csv'")
