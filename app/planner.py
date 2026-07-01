from google import genai
import re
from app.config import GEMINI_MODEL_NAME, GOOGLE_API_KEY

client = genai.Client(api_key=GOOGLE_API_KEY)

PLANNER_PROMPT = """
You are the Conversation Planner for an SHL Assessment Recommendation System.
Your ONLY task is to classify the latest user query into ONE of these labels:
clarify
recommend
compare
reject

Definitions:
clarify:
User wants an assessment but has not provided enough hiring details.

recommend:
User has provided enough hiring information to recommend SHL assessments.

compare:
User wants to compare two or more SHL assessments.

reject:
The question is unrelated to SHL assessments.

Return ONLY one word.
Never explain.
"""

VALID_LABELS = {"clarify", "recommend", "compare", "reject"}

# BUGFIX: "vs" was matching as a substring anywhere in the query
# (e.g. inside unrelated words). Word-boundaried now.
COMPARE_PATTERNS = [r"difference between", r"compare", r"\bvs\b", r"\bversus\b"]

REJECT_KEYWORDS = [
    "ipl",
    "cricket",
    "football",
    "weather",
    "movie",
    "recipe",
    "bitcoin",
    "stock",
    "music",
    "youtube",
]

RECOMMEND_KEYWORDS = [
    "hiring",
    "developer",
    "engineer",
    "manager",
    "analyst",
    "sales",
    "accountant",
    "graduate",
    "experience",
    "role",
    "backend",
    "frontend",
    "python",
    "java",
    "sql",
]

CLARIFY_KEYWORDS = [
    "assessment",
    "test",
    "recommend assessment",
    "need assessment",
    "help me choose",
]


def rule_based_planner(query: str):
    query = query.lower().strip()

    for pattern in COMPARE_PATTERNS:
        if re.search(pattern, query):
            return "compare"

    for word in REJECT_KEYWORDS:
        if word in query:
            return "reject"

    count = sum(word in query for word in RECOMMEND_KEYWORDS)
    if count >= 2:
        return "recommend"

    for word in CLARIFY_KEYWORDS:
        if word in query:
            return "clarify"

    return None


def llm_planner(query: str) -> str:
    prompt = f"{PLANNER_PROMPT}\n\nUser Query:\n{query}"
    response = client.models.generate_content(
    model=GEMINI_MODEL_NAME,
    contents=prompt
)

    label = response.text.strip().lower()

    # BUGFIX: Gemini could theoretically return something outside the
    # 4 valid labels (extra words, punctuation, etc). Previously this
    # would silently fall through to the "reject" branch in chat().
    # Defaulting to "clarify" is the safer failure mode.
    if label not in VALID_LABELS:
        return "clarify"
    return label


def planner(messages: list) -> str:
    """Hybrid planner: rule-based first, Gemini fallback. Same logic as notebook."""
    latest_query = messages[-1]["content"]
    decision = rule_based_planner(latest_query)
    if decision is not None:
        return decision
    return llm_planner(latest_query)
