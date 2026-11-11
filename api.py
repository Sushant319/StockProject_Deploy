
import os
import json
import logging
import hmac

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from stock_agent_updated import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-agent-api")

app = FastAPI(
    title="Enterprise Stock Analysis Agent",
    description="LangGraph multi-agent stock analysis API",
    version="1.1.0"
)

# API authentication
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False
)


def verify_api_key(api_key: str = Security(api_key_header)):
    expected_key = os.getenv("STOCK_API_KEY")

    if not expected_key:
        logger.error("STOCK_API_KEY is not configured")
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured"
        )

    if not api_key or not hmac.compare_digest(
        api_key, expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )

    return True


class StockRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Stock analysis question"
    )


@app.get("/")
def home():
    return {
        "message": "Stock Analysis Agent API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze_stock(
    request: StockRequest,
    authenticated: bool = Security(verify_api_key)
):
    try:
        logger.info("Received stock analysis request")

        initial_state = {
            "query": request.query,
            "results": []
        }

        result = await run_in_threadpool(
            graph.invoke,
            initial_state
        )

        response = {
            "ticker": result.get("ticker"),
            "tasks": result.get("tasks", []),
            "needs_clarification": result.get(
                "needs_clarification", False
            ),
            "clarification_question": result.get(
                "clarification_question"
            ),
            "results": result.get("results", []),
            "final_answer": result.get("final_answer")
        }

        safe_response = json.loads(
            json.dumps(response, default=str)
        )

        return JSONResponse(content=safe_response)

    except Exception:
        logger.exception("Stock analysis failed")
        raise HTTPException(
            status_code=500,
            detail="Stock analysis failed. Check application logs."
        )
