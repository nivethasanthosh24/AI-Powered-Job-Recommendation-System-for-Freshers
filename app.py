import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                    url("https://img.freepik.com/premium-vector/business-recruiting-employee-with-icon-technology-blue-background-office-staff-hiring-concept_252172-471.jpg");
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title
st.set_page_config(
    page_title="Fresher Job Recommendation",
    layout="wide"
)

st.title("🎯 AI-Powered Fresher Job Recommendation System")

st.markdown(
    "Find the best jobs based on your skills."
)

# Load dataset
data = pd.read_csv("final_recommended_jobs.csv")

# Fix missing values in skills column
data["skills"] = data["skills"].fillna("")

# User input
st.sidebar.header("Job Filters")

user_skills = st.sidebar.text_input(
    "Enter your skills"
)

# Job search by keyword
search_keyword = st.sidebar.text_input(
    "Search Job Title or Skill",
    key="search_keyword"
)

# Add experience filter
experience_filter = st.sidebar.selectbox(
    "Experience Level",
    ["All", "Fresher", "Experienced"],
    key="experience_filter"
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=1,
    max_value=100,
    value=10
)

locations = sorted(data["location"].dropna().unique())

selected_location = st.sidebar.selectbox(
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

        # Filter by search keyword
        if search_keyword:
            filtered_data = filtered_data[
                filtered_data["Job Title"].str.contains(
                    search_keyword,
                    case=False,
                    na=False
                )
            ]
        with st.spinner("Finding best job matches..."):
            top_jobs = filtered_data.head(top_n)

        # Experience filter
        if experience_filter != "All":
            filtered_data = filtered_data[
                filtered_data["job_type"] == experience_filter
            ]
        # Get top matches
        with st.spinner("Finding best job matches..."):
            top_jobs = filtered_data.head(top_n)

        # Show metric
        st.metric(
            label="Matched Jobs Found",
            value=len(top_jobs)
        )

        # Top company highlight
        if not top_jobs.empty:

            top_company = top_jobs["Company"].mode()[0]

            st.info(
                f"⭐ Top Hiring Company: {top_company}"
            )

        # Sort jobs
        top_jobs = filtered_data.sort_values(
            by="similarity_score",
            ascending=False
        )

        # Salary statistics

        import re

        if "Salary Range" in top_jobs.columns:

            st.subheader("Salary Overview")

            def extract_salary_avg(s):
                nums = re.findall(r'\d+', str(s))

                if len(nums) >= 2:
                    low = int(nums[0]) * 1000
                    high = int(nums[1]) * 1000
                    return (low + high) / 2

                elif len(nums) == 1:
                    return int(nums[0]) * 1000

                return None

            salary_data = top_jobs["Salary Range"].apply(extract_salary_avg).dropna()

            if not salary_data.empty:

                col1, col2, col3, col4 = st.columns(4)

                col1.metric("Jobs with Salary Info", len(salary_data))
                col2.metric("Average Salary", f"${int(salary_data.mean()):,}")
                col3.metric("Maximum Salary", f"${int(salary_data.max()):,}")
                col4.metric("Minimum Salary", f"${int(salary_data.min()):,}")

            else:
                st.warning("Salary data not available.")

            st.divider()

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