from app.config import TOP_K_COMPARE, TOP_K_RECOMMEND
from app.planner import client  # reuse the already-configured Gemini model
from app.config import TOP_K_COMPARE, TOP_K_RECOMMEND, GEMINI_MODEL_NAME
RECOMMENDATION_PROMPT = """
You are an SHL Assessment Recommendation Assistant.
You have received SHL assessments retrieved from the official catalog.

Rules:
1. Recommend ONLY from the provided assessments.
2. Never invent assessment names.
3. Never use outside knowledge.
4. Recommend the best 3-5 assessments (never fewer than 1, never more than 10).
5. Explain why each assessment matches the hiring requirement.
6. Mention job level if relevant.
7. Mention skills measured.

At the end write:
Final Recommendation
with bullet points.
"""

COMPARISON_PROMPT = """
You are an SHL Assessment Expert.
Compare ONLY the retrieved SHL assessments.

Rules:
1. Use ONLY the provided context.
2. Never use outside knowledge.
3. Compare:
   - Purpose
   - Skills Measured
   - Job Levels
   - Duration
   - Remote Testing
4. Finally tell which assessment is better depending on the user's request.

Give the comparison in a markdown table.
"""


def build_context(results):
    context = ""
    for i, item in enumerate(results, 1):
        context += f"""
==========================
Assessment {i}
Name:
{item['name']}
Description:
{item['description']}
Job Levels:
{", ".join(item['job_levels'])}
Skills / Categories:
{", ".join(item['keys'])}
Duration:
{item.get('duration', '')}
Remote:
{item.get('remote', '')}
Adaptive:
{item.get('adaptive', '')}
URL:
{item['url']}
==========================
"""
    return context


def recommendation_agent(retriever, user_query: str) -> dict:
    retrieved_docs = retriever.retrieve(user_query, top_k=TOP_K_RECOMMEND)

    if not retrieved_docs:
        return {
            "reply": "Sorry, I couldn't find any matching SHL assessments for this requirement.",
            "recommendations": [],
            "end_of_conversation": False,
        }

    context = build_context(retrieved_docs)
    prompt = f"{RECOMMENDATION_PROMPT}\n\nRetrieved SHL Assessments:\n{context}\n\nUser Query:\n{user_query}"
    response = client.models.generate_content(
    model=GEMINI_MODEL_NAME,
    contents=prompt
)

    return {
        "reply": response.text,
        # Recommendations list is built directly from retrieved catalog
        # items (not parsed out of the LLM's free text), so hallucinated
        # names/URLs can never leak into the structured response.
        "recommendations": [
            {"name": doc["name"], "url": doc["url"]} for doc in retrieved_docs
        ],
        "end_of_conversation": False,
    }


def comparison_agent(retriever, user_query: str) -> str:
    retrieved_docs = retriever.retrieve(user_query, top_k=TOP_K_COMPARE)

    if not retrieved_docs:
        return "Sorry, I couldn't find matching SHL assessments to compare."

    context = build_context(retrieved_docs)
    prompt = f"{COMPARISON_PROMPT}\n\nRetrieved SHL Assessments:\n{context}\n\nUser Query:\n{user_query}"
    response = client.models.generate_content(
    model=GEMINI_MODEL_NAME,
    contents=prompt
)
    return response.text
