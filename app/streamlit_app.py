import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =====================================================
# 1. PROJECT CONFIGURATION
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.pdf_parser import extract_text_from_pdf
from src.preprocessing import clean_text
from src.skill_extractor import extract_skills, DEFAULT_SKILLS

from src.matcher import (
    extract_job_skills,
    compare_skills,
    calculate_skill_coverage,
    calculate_overall_score,
)


# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Resume Screening and Job Matching",
    page_icon="📄",
    layout="wide",
)


# =====================================================
# 3. APPLICATION HEADER
# =====================================================

st.title("📄 Resume Screening and Job Matching")

st.markdown(
    """
    Upload candidate resumes and compare them against a job description.

    This application uses **Natural Language Processing (NLP)**,
    **TF-IDF text similarity**, and **skill matching** to help
    recruiters review candidate profiles.
    """
)

st.info(
    "This tool provides preliminary matching results to support human "
    "review. Scores should not be treated as definitive measures of "
    "candidate suitability or used as the sole basis for hiring decisions."
)


# =====================================================
# 4. JOB DESCRIPTION INPUT
# =====================================================

st.header("1. Enter Job Description")

job_description = st.text_area(
    "Paste the complete job description",
    height=240,
    placeholder=(
        "Example: We are looking for a Python developer with "
        "SQL, Machine Learning, Pandas, NumPy, and Data Analysis skills."
    ),
)


# =====================================================
# 5. RESUME UPLOAD
# =====================================================

st.header("2. Upload Candidate Resumes")

uploaded_files = st.file_uploader(
    "Upload one or more PDF resumes",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can select multiple PDF resumes at once.",
)


# =====================================================
# 6. PROCESS RESUMES
# =====================================================

if st.button("Analyze Resumes", type="primary"):

    if not job_description.strip():

        st.warning("Please enter a job description.")

    elif not uploaded_files:

        st.warning("Please upload at least one PDF resume.")

    else:

        required_skills = extract_job_skills(
            job_description,
            DEFAULT_SKILLS,
        )

        st.subheader("Required Skills")

        if required_skills:
            st.write(", ".join(required_skills))
        else:
            st.warning(
                "No skills from the configured skill dictionary were "
                "detected. Consider adding relevant skills to "
                "DEFAULT_SKILLS in src/skill_extractor.py."
            )

        cleaned_job = clean_text(job_description)

        # Store extracted resumes for shared TF-IDF comparison.
        candidate_documents = []
        candidate_data = []

        progress_bar = st.progress(0)

        status_text = st.empty()

        total_files = len(uploaded_files)

        # ---------------------------------------------
        # Extract text from each PDF
        # ---------------------------------------------

        for index, uploaded_file in enumerate(uploaded_files):

            status_text.write(
                f"Reading resume {index + 1} of {total_files}: "
                f"{uploaded_file.name}"
            )

            try:

                # Save uploaded PDF temporarily.
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False,
                ) as temp_file:

                    temp_file.write(uploaded_file.getvalue())
                    temp_path = Path(temp_file.name)

                try:
                    resume_text = extract_text_from_pdf(
                        str(temp_path)
                    )
                finally:
                    temp_path.unlink(missing_ok=True)

                if not resume_text or not resume_text.strip():

                    st.warning(
                        f"Could not extract text from "
                        f"{uploaded_file.name}. "
                        "The PDF may be scanned or image-only."
                    )

                    continue

                cleaned_resume = clean_text(resume_text)

                resume_skills = extract_skills(
                    resume_text,
                    DEFAULT_SKILLS,
                )

                candidate_documents.append(cleaned_resume)

                candidate_data.append(
                    {
                        "Resume": uploaded_file.name,
                        "Resume Skills": resume_skills,
                    }
                )

            except Exception as error:

                st.error(
                    f"Error processing {uploaded_file.name}: {error}"
                )

            progress_bar.progress(
                (index + 1) / total_files
            )

        # ---------------------------------------------
        # Calculate scores using one shared TF-IDF model
        # ---------------------------------------------

        results = []

        if candidate_documents:

            status_text.write("Calculating candidate matching scores...")

            try:

                # Fit the vectorizer on the job description
                # and all candidate resumes together.
                documents = [cleaned_job] + candidate_documents

                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    ngram_range=(1, 2),
                )

                document_vectors = vectorizer.fit_transform(
                    documents
                )

                job_vector = document_vectors[0:1]

                for index, candidate in enumerate(candidate_data):

                    resume_vector = document_vectors[index + 1:index + 2]

                    similarity = float(
                        cosine_similarity(
                            resume_vector,
                            job_vector,
                        )[0][0]
                    ) * 100

                    similarity = round(similarity, 2)

                    resume_skills = candidate["Resume Skills"]

                    comparison = compare_skills(
                        resume_skills,
                        required_skills,
                    )

                    matched_skills = comparison["matched_skills"]

                    missing_skills = comparison["missing_skills"]

                    coverage = calculate_skill_coverage(
                        matched_skills,
                        required_skills,
                    )

                    overall_score = calculate_overall_score(
                        similarity,
                        matched_skills,
                        required_skills,
                    )

                    results.append(
                        {
                            "Resume": candidate["Resume"],
                            "Text Similarity (%)": similarity,
                            "Skill Coverage (%)": coverage,
                            "Overall Score (%)": overall_score,
                            "Matched Skills": ", ".join(
                                matched_skills
                            ),
                            "Missing Skills": ", ".join(
                                missing_skills
                            ),
                        }
                    )

            except ValueError as error:

                st.error(
                    "Could not calculate text similarity. "
                    "Check whether the job description and resumes "
                    f"contain readable text. Details: {error}"
                )

            except Exception as error:

                st.error(
                    f"An error occurred during matching: {error}"
                )

        progress_bar.empty()
        status_text.empty()

        # =================================================
        # 7. DISPLAY CANDIDATE RANKINGS
        # =================================================

        if results:

            results_df = pd.DataFrame(results)

            results_df = results_df.sort_values(
                by="Overall Score (%)",
                ascending=False,
            ).reset_index(drop=True)

            results_df.insert(
                0,
                "Rank",
                range(1, len(results_df) + 1),
            )

            st.success(
                f"Analysis completed for {len(results_df)} resume(s)."
            )

            st.header("3. Candidate Rankings")

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
            )

            # ---------------------------------------------
            # Summary metrics
            # ---------------------------------------------

            st.header("Screening Summary")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Resumes Analyzed",
                len(results_df),
            )

            col2.metric(
                "Highest Matching Score",
                f"{results_df.iloc[0]['Overall Score (%)']:.2f}%",
            )

            col3.metric(
                "Required Skills",
                len(required_skills),
            )

            # =================================================
            # 8. CANDIDATE SCORE COMPARISON
            # =================================================

            st.header("4. Candidate Score Comparison")

            chart_df = results_df.set_index("Resume")[
                [
                    "Text Similarity (%)",
                    "Skill Coverage (%)",
                    "Overall Score (%)",
                ]
            ]

            # Separate bars make each metric easier to compare.
            st.bar_chart(
                chart_df,
                stack=False,
                height=400,
            )

            # =================================================
            # 9. CANDIDATE SKILL DETAILS
            # =================================================

            st.header("5. Candidate Skill Details")

            for _, row in results_df.iterrows():

                with st.expander(
                    f"Rank {row['Rank']}: {row['Resume']}"
                ):

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Overall Score",
                        f"{row['Overall Score (%)']:.2f}%",
                    )

                    col2.metric(
                        "Text Similarity",
                        f"{row['Text Similarity (%)']:.2f}%",
                    )

                    col3.metric(
                        "Skill Coverage",
                        f"{row['Skill Coverage (%)']:.2f}%",
                    )

                    st.markdown("**Matched Skills**")

                    if row["Matched Skills"]:
                        st.success(row["Matched Skills"])
                    else:
                        st.write("No matched skills detected.")

                    st.markdown("**Missing Skills**")

                    if row["Missing Skills"]:
                        st.warning(row["Missing Skills"])
                    else:
                        st.write("No missing required skills detected.")

            # =================================================
            # 10. DOWNLOAD RESULTS
            # =================================================

            st.header("6. Export Results")

            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="📥 Download Candidate Rankings (CSV)",
                data=csv_data,
                file_name="candidate_rankings.csv",
                mime="text/csv",
            )

        else:

            st.warning(
                "No resumes could be analyzed successfully. "
                "Please check the PDF files and try again."
            )


# =====================================================
# 11. FOOTER
# =====================================================

st.divider()

st.caption(
    "Resume Screening and Job Matching | "
    "Built with Python, Streamlit, spaCy, and scikit-learn."
)