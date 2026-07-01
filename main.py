from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.agents import comparison_agent, recommendation_agent
from app.catalog import build_texts, load_catalog
from app.planner import planner
from app.retriever import Retriever
from app.schemas import ChatRequest, ChatResponse, HealthResponse

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="RAG-based SHL assessment recommender using FAISS retrieval + Gemini generation.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state, built once on startup.
catalog = None
retriever: Retriever | None = None


@app.on_event("startup")
def startup_event():
    global catalog, retriever
    catalog = load_catalog()
    texts = build_texts(catalog)
    retriever = Retriever(catalog, texts)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if retriever is None:
        raise HTTPException(status_code=503, detail="Service is still starting up.")

    try:
        messages = request.to_messages()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    latest_query = messages[-1]["content"]
    decision = planner(messages)

    if decision == "clarify":
        return ChatResponse(
            reply="Which role are you hiring for? Please share the role, experience level, and skills required.",
            recommendations=[],
            end_of_conversation=False,
        )

    if decision == "recommend":
        result = recommendation_agent(retriever, latest_query)
        return ChatResponse(**result)

    if decision == "compare":
        reply = comparison_agent(retriever, latest_query)
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    # decision == "reject"
    # NOTE: notebook had end_of_conversation=True here. Changed to False —
    # refusing one off-topic question shouldn't necessarily end the whole
    # session. Flip back to True if the assignment spec requires it.
    return ChatResponse(
        reply="Sorry, I can only answer questions related to SHL assessments.",
        recommendations=[],
        end_of_conversation=False,
    )
