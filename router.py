import uuid

import anthropic
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from service import start_reply

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: str | None = None


@router.post("/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    conversation_id = body.conversation_id or str(uuid.uuid4())

    try:
        chunks = await start_reply(conversation_id, body.message)
    except anthropic.RateLimitError:
        raise HTTPException(429, "Claude is rate limited, retry shortly")
    except anthropic.BadRequestError as e:
        raise HTTPException(400, e.message)
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Claude API error ({e.status_code})")
    except anthropic.APIConnectionError:
        raise HTTPException(503, "Could not reach the Claude API")

    return StreamingResponse(
        chunks,
        media_type="text/plain; charset=utf-8",
        headers={
            "X-Conversation-Id": conversation_id,
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stop nginx-style proxies buffering
        },
    )
