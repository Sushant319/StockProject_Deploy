
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
import json
import logging

from stock_agent_updated import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("stock-agent-api")

app = FastAPI(
    title="Enterprise Stock Analysis Agent",
    description="LangGraph multi-agent stock analysis API",
    version="1.0.0"
)


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
async def analyze_stock(request: StockRequest):

    try:
        logger.info("Received stock query: %s", request.query)

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

        # Convert NumPy/Pandas values to JSON-safe representations.
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
