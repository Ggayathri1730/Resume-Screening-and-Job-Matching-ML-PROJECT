import re
from typing import List, Optional

DEFAULT_SKILLS = [
    "python",
    "java",
    "c++",
    "sql",
    "machine learning",
    "deep learning",
    "natural language processing",
    "nlp",
    "tensorflow",
    "pytorch",
    "scikit-learn",
    "pandas",
    "numpy",
    "spacy",
    "streamlit",
    "fastapi",
    "git",
    "github",
    "docker",
    "aws",
    "azure",
    "data analysis",
    "data visualization",
    "excel",
    "power bi",
    "tableau",
    "flask",
    "html",
    "css",
    "javascript",
    "matplotlib",
    "statistics",
    "data cleaning",
    "data preprocessing",
    "deep learning",
    "communication",
    "leadership",
    "problem solving",
    "project management",
    "teamwork"
]


def extract_skills(
    text: str,
    skills: Optional[List[str]] = None
) -> List[str]:

    text_lower = (text or "").lower()

    skills_to_check = skills if skills is not None else DEFAULT_SKILLS

    found = []

    for skill in skills_to_check:
        skill = skill.strip()

        if not skill:
            continue

        pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"

        if re.search(pattern, text_lower):
            found.append(skill)

    return sorted(set(found), key=str.lower)


def extract_ner(text: str, nlp):
    doc = nlp(text or "")

    return [
        {
            "text": ent.text,
            "label": ent.label_
        }
        for ent in doc.ents
    ]


def extract_basic_info(text: str, nlp):
    entities = extract_ner(text, nlp)

    return {
        "persons": [
            entity["text"]
            for entity in entities
            if entity["label"] == "PERSON"
        ],
        "organizations": [
            entity["text"]
            for entity in entities
            if entity["label"] == "ORG"
        ],
        "locations": [
            entity["text"]
            for entity in entities
            if entity["label"] in {"GPE", "LOC"}
        ]
    }