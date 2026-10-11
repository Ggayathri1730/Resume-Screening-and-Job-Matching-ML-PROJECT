import re
from typing import List, Dict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# 1. NORMALIZE SKILL NAMES
# ==================================================

def normalize_skill(skill: str) -> str:
    """Convert a skill name to lowercase and normalize spaces."""

    return re.sub(r"\s+", " ", skill.strip().lower())


# ==================================================
# 2. CALCULATE TEXT SIMILARITY
# ==================================================

def calculate_similarity(
    resume_text: str,
    job_description: str
) -> float:
    """
    Calculate TF-IDF cosine similarity between a resume
    and a job description.

    Returns a percentage between 0 and 100.
    """

    if not resume_text or not job_description:
        return 0.0

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:
        vectors = vectorizer.fit_transform(
            [resume_text, job_description]
        )

        similarity = cosine_similarity(
            vectors[0:1],
            vectors[1:2]
        )[0][0]

        return round(float(similarity) * 100, 2)

    except ValueError:
        return 0.0


# ==================================================
# 3. EXTRACT REQUIRED SKILLS FROM JOB DESCRIPTION
# ==================================================

def extract_job_skills(
    job_description: str,
    skill_list: List[str]
) -> List[str]:
    """
    Extract skills from a job description using a
    predefined skill list.
    """

    found_skills = set()
    text = (job_description or "").lower()

    for skill in skill_list:
        normalized = normalize_skill(skill)

        if not normalized:
            continue

        pattern = (
            r"(?<!\w)"
            + re.escape(normalized)
            + r"(?!\w)"
        )

        if re.search(pattern, text):
            found_skills.add(normalized)

    return sorted(found_skills)


# ==================================================
# 4. COMPARE RESUME SKILLS WITH REQUIRED SKILLS
# ==================================================

def compare_skills(
    resume_skills: List[str],
    required_skills: List[str]
) -> Dict[str, List[str]]:
    """
    Identify matched skills and missing skills.
    """

    resume_set = {
        normalize_skill(skill)
        for skill in resume_skills
        if skill and skill.strip()
    }

    required_set = {
        normalize_skill(skill)
        for skill in required_skills
        if skill and skill.strip()
    }

    matched_skills = sorted(
        resume_set.intersection(required_set)
    )

    missing_skills = sorted(
        required_set.difference(resume_set)
    )

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }


# ==================================================
# 5. CALCULATE SKILL COVERAGE
# ==================================================

def calculate_skill_coverage(
    matched_skills: List[str],
    required_skills: List[str]
) -> float:
    """
    Calculate the percentage of required skills
    present in the resume.
    """

    required_set = {
        normalize_skill(skill)
        for skill in required_skills
        if skill and skill.strip()
    }

    matched_set = {
        normalize_skill(skill)
        for skill in matched_skills
        if skill and skill.strip()
    }

    if not required_set:
        return 0.0

    matched_count = len(
        required_set.intersection(matched_set)
    )

    coverage = (
        matched_count / len(required_set)
    ) * 100

    return round(coverage, 2)


# ==================================================
# 6. CALCULATE OVERALL MATCHING SCORE
# ==================================================

def calculate_overall_score(
    similarity_score: float,
    matched_skills: List[str],
    required_skills: List[str]
) -> float:
    """
    Calculate the overall matching score.

    Text similarity contributes 70%.
    Skill coverage contributes 30%.
    """

    similarity_score = max(
        0.0,
        min(100.0, float(similarity_score))
    )

    skill_coverage = calculate_skill_coverage(
        matched_skills,
        required_skills
    )

    overall_score = (
        0.70 * similarity_score
        + 0.30 * skill_coverage
    )

    return round(overall_score, 2)