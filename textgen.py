import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Page configuration
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="🤖"
)

st.title("🤖 AI Text Generation using LSTM")

st.write("""
Enter a starting text, choose the number of words you want to generate,
and let the LSTM model continue the sentence.
""")

# Training text
text = """
python is a programming language
python is easy to learn
python is powerful for machine learning
machine learning uses python
machine learning is interesting
machine learning learns from data
deep learning uses neural networks
deep learning is powerful
"""

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

total_words = len(tokenizer.word_index) + 1

# Create input sequences
sequences = []

for line in text.strip().split("\n"):
    tokens = tokenizer.texts_to_sequences([line])[0]

    for i in range(1, len(tokens)):
        sequence = tokens[:i + 1]
        sequences.append(sequence)

# Padding
max_len = max(len(sequence) for sequence in sequences)

sequences = pad_sequences(
    sequences,
    maxlen=max_len,
    padding="pre"
)

# Input and output
x = sequences[:, :-1]
y = sequences[:, -1]

# LSTM model
model = Sequential([
    Embedding(
        input_dim=total_words,
        output_dim=20,
        input_length=max_len - 1
    ),

    LSTM(100),

    Dense(
        total_words,
        activation="softmax"
    )
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train model
with st.spinner("Training the LSTM model..."):
    model.fit(
        x,
        y,
        epochs=300,
        verbose=0
    )

# User input
user_text = st.text_input(
    "Enter starting text:",
    placeholder="Example: machine learning"
)

# Number of words
num_words = st.slider(
    "How many words do you want to generate?",
    min_value=1,
    max_value=10,
    value=5
)

# Text generation function
def generate_text(seed_text, num_words):

    generated_text = seed_text

    for _ in range(num_words):

        tokens = tokenizer.texts_to_sequences(
            [generated_text]
        )[0]

        tokens = pad_sequences(
            [tokens],
            maxlen=max_len - 1,
            padding="pre"
        )

        prediction = model.predict(
            tokens,
            verbose=0
        )

        predicted_word_id = np.argmax(prediction)

        predicted_word = ""

        for word, index in tokenizer.word_index.items():

            if index == predicted_word_id:
                predicted_word = word
                break

        generated_text += " " + predicted_word

    return generated_text


# Generate text
if user_text.strip() == "":

    st.warning(
        "Please enter some starting text."
    )

else:

    with st.spinner("Generating text..."):

        result = generate_text(
            user_text,
            num_words
        )

    st.success("Text Generated Successfully!")

    st.subheader("Generated Text")

    st.info(result)