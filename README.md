# SHL Assessment Recommendation API

RAG-based SHL assessment recommender: **FAISS retrieval + Gemini 2.5 Flash generation**,
served via FastAPI.

## Architecture (unchanged from prototype)

```
User
  ↓
Hybrid Planner (rule-based → Gemini fallback)
  ↓
Retriever (FAISS, SentenceTransformer embeddings)
  ↓
Context Builder
  ↓
Gemini 2.5 Flash
  ↓
Recommendation / Comparison Response
```

## Project structure

```
shl-assessment-recommender/
├── main.py                # FastAPI app, /health and /chat routes
├── requirements.txt
├── .gitignore
├── data/
│   └── shl_catalog_fixed.json   # <-- put your repaired catalog JSON here
├── cache/                  # auto-generated FAISS index + embeddings (gitignored)
└── app/
    ├── config.py            # env-driven settings
    ├── catalog.py            # catalog loading + text building
    ├── retriever.py           # SentenceTransformer + FAISS wrapper
    ├── planner.py             # hybrid rule-based + Gemini intent planner
    ├── agents.py              # recommendation_agent, comparison_agent
    └── schemas.py             # pydantic request/response models
```

### Why split into these files?
- `catalog.py` / `retriever.py` / `planner.py` / `agents.py` are separated because
  each has a single responsibility and gets imported independently by `main.py`.
  This also makes each piece unit-testable on its own (e.g. you can test
  `rule_based_planner()` without touching FAISS or Gemini at all).
- `schemas.py` is separate so the API contract (request/response shape) is
  defined once and reused for validation + auto-generated OpenAPI docs.
- Your original logic (prompts, retrieval math, planner rules, chat flow) is
  preserved as-is — only moved into files, plus the specific bug fixes noted below.

## What changed vs. your notebook

1. **Bugfix:** `retrieve()` now also returns `duration`, `remote`, `adaptive`,
   `languages` — these were missing before, which is why your comparison output
   said "Duration & Remote Testing: Not specified in the context." even though
   the data exists in the catalog.
2. **Bugfix:** compare-intent regex `"vs"` now uses `\bvs\b` word boundary to
   avoid matching substrings inside unrelated words.
3. **Safety fix:** `llm_planner()` now validates Gemini's output is one of the
   4 expected labels, defaulting to `"clarify"` if not (previously an
   unexpected response would silently fall into the reject branch).
4. **Behavior change:** `end_of_conversation` on reject is now `False` instead
   of `True` — refusing one off-topic message doesn't end the whole session.
   Revert this one line in `main.py` if your assignment spec requires `True`.
5. **Performance:** FAISS index + embeddings are cached to disk on first
   startup (`cache/` folder) so subsequent restarts don't re-embed all catalog
   items every time.
6. **New:** `GET /health` and `POST /chat` FastAPI endpoints, matching your
   exact response schema.

Everything else — prompts, planner rules, retrieval logic, agent logic — is
untouched.

## API

### `GET /health`
```json
{ "status": "ok" }
```

### `POST /chat`
Request (either form works):
```json
{ "message": "Hiring Java Backend Developer with communication skills" }
```
or, for full conversation history:
```json
{ "messages": [{ "role": "user", "content": "Hiring Java Backend Developer" }] }
```

Response:
```json
{
  "reply": "...",
  "recommendations": [
    { "name": "Java 8 (New)", "url": "https://www.shl.com/products/product-catalog/view/java-8-new/" }
  ],
  "end_of_conversation": false
}
```

## Local setup

```bash
git clone <your-repo-url>
cd shl-assessment-recommender

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt

# put your repaired catalog JSON at data/shl_catalog_fixed.json

export GOOGLE_API_KEY="your_gemini_api_key"

uvicorn main:app --reload --port 8000
```

Test:
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hiring Java Backend Developer with communication skills"}'
```

## Deployment guide (Render)

1. **Push this project to GitHub** (make sure `data/shl_catalog_fixed.json`
   is committed — it's not gitignored).

2. **Create a new Web Service on Render**
   - Connect your GitHub repo.
   - Environment: `Python 3`.
   - Build Command:
     ```
     pip install -r requirements.txt
     ```
   - Start Command:
     ```
     uvicorn main:app --host 0.0.0.0 --port $PORT
     ```

3. **Set environment variables** (Render dashboard → Environment):
   - `GOOGLE_API_KEY` = your Gemini API key
   - (optional) `GEMINI_MODEL_NAME` = `gemini-2.5-flash`
   - (optional) `TOP_K_RECOMMEND` = `5`
   - (optional) `TOP_K_COMPARE` = `3`

4. **Instance type:** free tier works, but first request after a cold start
   will be slower (loading SentenceTransformer + building/loading FAISS
   index). Cache persistence on Render's free tier is not guaranteed across
   deploys (ephemeral disk), so the embeddings may rebuild once per deploy —
   this is fine for 377 catalog items (a few seconds), just not instant.

5. **Verify deployment:**
   ```bash
   curl https://<your-app>.onrender.com/health
   ```

6. **Note on `google-generativeai`:** this package shows a deprecation
   warning in your notebook output (Google is moving everyone to
   `google-genai`). It still works and is what your prototype uses, so it's
   kept as-is here per your instructions not to change frameworks — but keep
   an eye on it, since deprecated packages can eventually stop receiving
   security/API updates.

## Known limitations to flag to your evaluator

- `TOP_K_COMPARE=3` retrieves up to 3 items for comparison; if a user asks to
  compare more than 3 named assessments, only the top 3 by similarity will be
  included in context.
- The planner's rule-based layer is keyword-based, not semantic. Genuinely
  ambiguous phrasing falls through to the Gemini fallback correctly, but edge
  cases with unusual phrasing may still get misclassified — this is inherent
  to the rule-based-first design, not a new issue introduced here.
