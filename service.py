from collections.abc import AsyncIterator
from contextlib import AsyncExitStack

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.AsyncAnthropic()

MODEL = "claude-haiku-5-5"

# conversation_id -> message history. In-memory only: lost on restart and
# not shared across workers.
conversations: dict[str, list[dict]] = {}


async def start_reply(conversation_id: str, text: str) -> AsyncIterator[str]:
    """Open a Claude stream and return an iterator over its text chunks.

    The HTTP request to Claude is made here, before any bytes go to the client,
    so auth/rate-limit/validation errors raise and can become proper HTTP
    errors. The user's message is saved right away; Claude's reply is saved
    once it completes.
    """
    history = conversations.setdefault(conversation_id, [])
    history.append({"role": "user", "content": text})

    stack = AsyncExitStack()
    stream = await stack.enter_async_context(
        client.messages.stream(
            model=MODEL,
            max_tokens=16000,
            messages=history,
        )
    )

    async def chunks() -> AsyncIterator[str]:
        parts = []
        try:
            async for chunk in stream.text_stream:
                parts.append(chunk)
                yield chunk

            final = await stream.get_final_message()
            if final.stop_reason == "refusal":
                yield "\n[Claude declined to respond]"
                return

            history.append({"role": "assistant", "content": "".join(parts)})
        except anthropic.APIError as e:
    
            yield f"\n[stream error: {e.message}]"
        finally:
            await stack.aclose()

    return chunks()
