import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import logging
from surprise import SVD

# Set up logging configuration
logging.basicConfig(level=logging.DEBUG)  # Set the logging level to DEBUG

# Use environment variables for file paths
MODEL_FILE = os.environ.get('MODEL_FILE', 'model.pkl')
BOOK_PIVOT_FILE = os.environ.get('BOOK_PIVOT_FILE', 'book_pivot.pkl')
MODEL_DF_FILE = os.environ.get('MODEL_DF_FILE', 'final_df.pkl')
BOOK_NAMES_FILE = os.environ.get('BOOK_NAMES_FILE', 'book_names.pkl')


def load_model_data():
    try:
        logging.info("Loading model data...")

        # Load the SVD model using joblib
        model = joblib.load(MODEL_FILE)
        if not isinstance(model, SVD):  # Check model type
            raise ValueError("Invalid model format")

        # Load other necessary data
        book_pivot = joblib.load(BOOK_PIVOT_FILE)
        final_df = joblib.load(MODEL_DF_FILE)

        with open(BOOK_NAMES_FILE, 'rb') as f:
            book_names = joblib.load(f)

        logging.info("Model data loaded successfully.")
        return model, book_pivot, final_df, book_names, 200  # Success
    except FileNotFoundError as e:
        logging.error(f"File not found error: {e}")
        return None, None, None, None, 404  # File not found
    except Exception as e:
        logging.error(f"Error loading model data: {e}")
        return None, None, None, None, 500  # Internal server error


def recommend_book(book_name, model, book_pivot, final_df):
    """Define a function to recommend books based on a given book name."""
    recommendations = []

    # Check if the input book name is in the book_pivot index
    if book_name in book_pivot.index:

        # Find the index of the input book name in the book_pivot index array
        book_id = book_pivot.index.get_loc(book_name)

        # Predict ratings for all users for the internal book ID
        predicted_ratings = [model.predict(str(user_id), str(book_id)).est for user_id in range(1, len(book_pivot) + 1)]

        # Find indices of top 5 books with the highest predicted ratings
        top_indices = np.argsort(predicted_ratings)[::-1][:6]

        # Extract recommended book names and their URLs
        for idx in top_indices:
            book_title = book_pivot.index[idx]
            if book_title != book_name:  # Exclude the input book itself
                # Assuming index is 1-based
                book_url = final_df.loc[final_df['Book-Title'] == book_title, 'Image-URL-M'].values[0]
                book_author = final_df.loc[final_df['Book-Title'] == book_title, 'Book-Author'].values[0]
                book_year = final_df.loc[final_df['Book-Title'] == book_title, 'Year-Of-Publication'].values[0]
                recommendations.append(
                    {"title": book_title, "url": book_url, "author": book_author, "year": book_year})

    return recommendations


# Load model data
model, book_pivot, final_df, _, status_code = load_model_data()

# Streamlit app layout
st.title("Book Recommendation System")

# Input column
col1, col2 = st.columns(2)
with col1:
    book_name = st.text_input("Enter a book name:")
with col2:
    if st.button("Recommend"):
        if status_code == 200:
            recommendations = recommend_book(book_name, model, book_pivot, final_df)
            if recommendations:
                st.write("**Recommended Books:**")
                for rec in recommendations:
                    st.write(f"- **Title:** {rec['title']}")
                    st.write(f"  **Author:** {rec['author']}")
                    st.write(f"  **Year of Publication:** {rec['year']}")
                    st.write("  **Summary:**")
                    st.image(rec['url'], caption=rec['title'], use_column_width=True)
            else:
                st.write("No recommendations found.")
        else:
            st.error("Error loading model data. Please check the logs.")

