# Resume Screening and Job Matching Using Machine Learning

## 1. Project Overview

Resume Screening and Job Matching is a machine learning and
natural language processing project that compares resumes
with a given job description.

The system extracts resume text, identifies relevant skills,
calculates text similarity, and ranks candidates according
to their match with the job requirements.

## 2. Objectives

- Automate preliminary resume screening.
- Extract relevant skills from resumes.
- Compare resumes with job descriptions.
- Calculate similarity and skill coverage scores.
- Rank candidates based on their overall matching scores.
- Display results through an interactive web application.

## 3. Technologies Used

- Python
- Machine Learning
- Natural Language Processing (NLP)
- spaCy Named Entity Recognition (NER)
- TF-IDF Vectorization
- Cosine Similarity
- Scikit-learn
- Pandas
- Streamlit
- PyMuPDF or the PDF parsing library used by the project

## 4. Main Features

- Upload multiple PDF resumes.
- Enter a job description.
- Extract text from resumes.
- Identify relevant skills.
- Calculate text similarity.
- Compare matched and missing skills.
- Calculate overall matching scores.
- Rank candidates.
- Visualize candidate scores.
- Download screening results as a CSV file.

## 5. Project Structure

    resume-screening-job-matching/
    ├── app/
    │   └── streamlit_app.py
    ├── data/
    ├── models/
    ├── notebooks/
    ├── results/
    ├── src/
    ├── .gitignore
    ├── README.md
    └── requirements.txt

## 6. How to Run the Project

### Step 1: Clone or download the project

Open the project folder in VS Code.

### Step 2: Activate the virtual environment

Windows PowerShell:

    venv\Scripts\activate

### Step 3: Install dependencies

    python -m pip install -r requirements.txt

### Step 4: Install the spaCy English model

    python -m spacy download en_core_web_sm

### Step 5: Run the application

    python -m streamlit run app/streamlit_app.py

### Step 6: Use the application

1. Enter a job description.
2. Upload PDF resumes.
3. Click Analyze Resumes.
4. Review candidate rankings and scores.
5. Examine matched and missing skills.
6. Download the results if required.

## 7. Scoring Method

The system combines text similarity and skill coverage.

Overall Score =
(0.70 × Text Similarity) +
(0.30 × Skill Coverage)

Text similarity is calculated using TF-IDF and cosine
similarity. Skill coverage measures the proportion of
identified required skills found in a resume.

## 8. Limitations

- Skill extraction depends on the configured skill list.
- PDF text extraction may not work well with scanned resumes.
- Similarity scores do not guarantee candidate suitability.
- The system does not replace human review.
- The scoring method requires further evaluation on
  representative real-world data.

## 9. Conclusion

This project demonstrates how machine learning and natural
language processing can support preliminary resume screening
by comparing resumes with job descriptions and presenting
ranked results to recruiters.

The scores are decision-support indicators and should not
be used as the sole basis for hiring decisions.