import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page title
st.set_page_config(
    page_title="Fresher Job Recommendation",
    layout="wide"
)

st.title("🎯 AI-Powered Fresher Job Recommendation System")

st.markdown(
    "Find the best jobs based on your skills using Machine Learning."
)

# Load dataset
data = pd.read_csv("final_recommended_jobs.csv")

# User input
st.sidebar.header("Job Filters")

user_skills = st.sidebar.text_input(
    "Enter your skills"
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    1,
    20,
    10
)

locations = sorted(data["location"].dropna().unique())

selected_location = st.sidebar.selectbox(
    "Select Location",
    ["All Locations"] + list(locations)
)


# Location filter
locations = sorted(data["location"].dropna().unique())

selected_location = st.selectbox(
    "Select Location",
    ["All Locations"] + list(locations)
)

# Button
if st.button("Recommend Jobs"):

    if user_skills:

        # Convert skills to lowercase
        data["skills"] = data["skills"].str.lower()

        # Create vectorizer
        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        # Convert job skills to vectors
        job_vectors = vectorizer.fit_transform(
            data["skills"]
        )

        # Convert user skills to vector
        user_vector = vectorizer.transform(
            [user_skills]
        )

        # Calculate similarity
        similarity_scores = cosine_similarity(
            user_vector,
            job_vectors
        )

        # Add similarity score
        data["similarity_score"] = similarity_scores.flatten()

        # Filter by location
        if selected_location != "All Locations":
            filtered_data = data[
                data["location"] == selected_location
            ]
        else:
            filtered_data = data

        # Sort jobs
        top_jobs = filtered_data.sort_values(
            by="similarity_score",
            ascending=False
        )

        # Show results
        st.subheader("Top Recommended Jobs")

        st.dataframe(
            top_jobs[
                [
                    "Job Title",
                    "Company",
                    "similarity_score"
                ]
            ].head(top_n)
        )

        # Download button
        csv = top_jobs.head(top_n).to_csv(
            index=False
        )

        st.download_button(
            label="Download Recommended Jobs",
            data=csv,
            file_name="recommended_jobs.csv",
            mime="text/csv"
        )

    else:

        st.warning("Please enter your skills.")