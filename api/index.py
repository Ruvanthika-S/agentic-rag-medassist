from functools import lru_cache
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scripts.agents.orchestrator import Orchestrator


load_dotenv()

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


@lru_cache(maxsize=1)
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


app = FastAPI(title="MedAssist API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.post("/api/ask")
async def ask(request: AskRequest) -> dict:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    try:
        return await run_in_threadpool(get_orchestrator().run, question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("MedAssist pipeline failed for question %r", question)
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc