import os
import re
import time
import string
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Set
from collections import defaultdict, Counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import openai

# FastAPI app
app = FastAPI(title="Advanced Flashcard API", version="2.2.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class FlashcardRequest(BaseModel):
    content: str = Field(..., min_length=100, max_length=10000)
    subject: Optional[str] = None
    max_flashcards: Optional[int] = Field(50, ge=10, le=100)
    difficulty_level: Optional[str] = Field("medium")

class FlashcardResponse(BaseModel):
    question: str
    answer: str
    type: str
    subject: str
    difficulty: str
    confidence_score: float

class GenerationResult(BaseModel):
    flashcards: List[FlashcardResponse]
    metadata: Dict[str, Any]
    performance: Dict[str, float]
    success: bool
    message: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

# Generator instance
from flashcard_generator import AdvancedFlashcardGenerator as BaseGenerator

class HybridFlashcardGenerator(BaseGenerator):
    def __init__(self):
        super().__init__()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key

    def generate_llm_flashcards(self, content: str, max_cards: int = 5) -> List[Dict]:
        if not self.openai_api_key:
            return []  # Skip LLM generation if key is not set

        prompt = (
            f"Generate {max_cards} educational flashcards from the text below. "
            f"Each flashcard should include a question and answer.\n\nText:\n{content[:4000]}"
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=1000,
            )
            text = response.choices[0].message.content
            cards = []
            for match in re.findall(r"Question:\s*(.*?)\nAnswer:\s*(.*?)\n", text, re.DOTALL):
                question, answer = match
                cards.append({
                    "type": "llm",
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "confidence_score": 0.95,
                    "educational_value": 0.9
                })
            return cards
        except Exception as e:
            print("LLM generation error:", e)
            return []

    def generate_flashcards(self, content: str, subject: Optional[str] = None,
                            max_flashcards: int = 50, difficulty: str = "medium") -> Dict:
        base_result = super().generate_flashcards(content, subject, max_flashcards, difficulty)

        if not base_result["success"]:
            return base_result

        try:
            llm_cards = self.generate_llm_flashcards(content, max_cards=5)
            for card in llm_cards:
                card["subject"] = subject or base_result["metadata"].get("subject", "general")
                card["difficulty"] = difficulty

            base_result["flashcards"].extend(llm_cards)
            base_result["metadata"]["llm_cards"] = len(llm_cards)
            base_result["metadata"]["final_count"] += len(llm_cards)
            base_result["message"] += f" (plus {len(llm_cards)} enhanced via OpenAI)"
        except Exception as e:
            base_result["message"] += f" (OpenAI fallback due to error: {str(e)})"

        return base_result


# Use hybrid generator
generator = HybridFlashcardGenerator()

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.2.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.2.0"
    )

@app.post("/generate-flashcards", response_model=GenerationResult)
async def generate_flashcards(request: FlashcardRequest):
    try:
        if len(request.content) < 100:
            raise HTTPException(status_code=400, detail="Content too short")

        result = generator.generate_flashcards(
            content=request.content,
            subject=request.subject,
            max_flashcards=request.max_flashcards,
            difficulty=request.difficulty_level
        )

        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])

        flashcard_responses = [
            FlashcardResponse(**card) for card in result['flashcards']
        ]

        return GenerationResult(
            flashcards=flashcard_responses,
            metadata=result['metadata'],
            performance=result['performance'],
            success=result['success'],
            message=result['message']
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/subjects")
async def get_subjects():
    return {"subjects": list(generator.subjects.keys()), "default": "general"}

@app.get("/difficulty-levels")
async def get_difficulty_levels():
    return {"levels": ["easy", "medium", "hard"], "default": "medium"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
