from typing import List, Optional

from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    # Preferred: full conversation history, matches the notebook's
    # chat(messages) signature exactly.
    messages: Optional[List[Message]] = None
    # Convenience: single-message stateless calls without wrapping in a list.
    message: Optional[str] = None

    def to_messages(self) -> List[dict]:
        if self.messages:
            return [m.dict() for m in self.messages]
        if self.message:
            return [{"role": "user", "content": self.message}]
        raise ValueError("Either 'messages' or 'message' must be provided.")


class Recommendation(BaseModel):
    name: str
    url: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


class HealthResponse(BaseModel):
    status: str
