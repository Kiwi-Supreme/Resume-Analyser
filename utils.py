import re
import io

import docx
from pypdf import PdfReader

SKILLS = [
    "python", "java", "javascript", "c++", "c#", "sql", "r", "scala", "html", "css",
    "react", "angular", "node", "django", "flask", "spring", "hibernate",
    "machine learning", "deep learning", "nlp", "data science", "data analysis",
    "tensorflow", "keras", "pytorch", "scikit-learn", "pandas", "numpy",
    "tableau", "power bi", "excel", "hadoop", "spark", "etl", "data warehouse",
    "aws", "azure", "docker", "kubernetes", "git", "jenkins", "linux", "devops",
    "selenium", "testing", "sap", "recruitment", "payroll", "employee relations",
    "accounting", "auditing", "financial reporting", "sales", "marketing",
    "customer service", "project management", "agile", "scrum", "autocad",
    "photoshop", "teaching", "training", "nutrition", "fitness", "compliance",
]


def extract_text(filename, data):
    """Return raw text from PDF / DOCX / TXT bytes."""
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return " ".join((page.extract_text() or "") for page in reader.pages)
    if name.endswith(".docx"):
        document = docx.Document(io.BytesIO(data))
        return " ".join(p.text for p in document.paragraphs)
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError("Unsupported file type. Use PDF, DOCX or TXT.")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z+#\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_skills(text):
    text = " " + clean_text(text) + " "
    found = []
    for skill in SKILLS:
        if re.search(r"(?<![a-z])" + re.escape(skill) + r"(?![a-z])", text):
            found.append(skill)
    return found
