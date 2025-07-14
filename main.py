import os
import re
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import defaultdict, Counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import openai

# FastAPI app
app = FastAPI(title="Intelligent Flashcard API", version="3.0.0")

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
    content: str = Field(..., min_length=100, max_length=30000)
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

# Generator class
class SmartFlashcardGenerator:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            openai.api_key = self.openai_api_key

    def generate_with_openai(self, content: str, max_cards: int = 10) -> List[Dict]:
        if not self.openai_api_key:
            return []

        prompt = (
            f"Generate {max_cards} highly educational, exam-focused flashcards from the following text. "
            f"Each flashcard must have a concise and clear question and answer. Format as:\n"
            f"Question: ...\nAnswer: ...\n\nText:\n{content[:6000]}"
        )
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1500,
            )
            text = response.choices[0].message.content
            cards = []
            for match in re.findall(r"Question:\s*(.*?)\nAnswer:\s*(.*?)\n", text, re.DOTALL):
                question, answer = match
                cards.append({
                    "type": "llm",
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "confidence_score": 0.95
                })
            return cards
        except Exception as e:
            print("OpenAI error:", e)
            return []

    def generate_rule_based(self, content: str, max_cards: int = 20) -> List[Dict]:
        sentences = re.split(r'(?<=[.!?])\s+', content)
        cards = []
        for sentence in sentences:
            if ' is ' in sentence and len(sentence.split()) > 5:
                parts = sentence.split(' is ', 1)
                question = f"What is {parts[0].strip()}?"
                answer = parts[1].strip().rstrip('.').capitalize()
                cards.append({
                    "type": "definition",
                    "question": question,
                    "answer": answer,
                    "confidence_score": 0.7
                })
            if len(cards) >= max_cards:
                break
        return cards

    def generate_flashcards(self, content: str, subject: Optional[str] = None, max_flashcards: int = 50, difficulty: str = "medium") -> Dict:
        start = time.time()
        rule_cards = self.generate_rule_based(content, max_cards=max_flashcards // 2)
        llm_cards = self.generate_with_openai(content, max_cards=max_flashcards - len(rule_cards))

        flashcards = rule_cards + llm_cards
        for card in flashcards:
            card['subject'] = subject or "general"
            card['difficulty'] = difficulty

        return {
            "flashcards": flashcards,
            "metadata": {
                "subject": subject or "general",
                "rule_based": len(rule_cards),
                "llm_based": len(llm_cards),
                "final_count": len(flashcards)
            },
            "performance": {
                "processing_time": time.time() - start,
                "flashcards_per_second": len(flashcards) / max(time.time() - start, 0.1)
            },
            "success": True,
            "message": f"Generated {len(flashcards)} flashcards including AI-refined ones"
        }

# Initialize generator
generator = SmartFlashcardGenerator()

@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="healthy", timestamp=datetime.now().isoformat(), version="3.0.0")

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy", timestamp=datetime.now().isoformat(), version="3.0.0")

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

        flashcard_responses = [FlashcardResponse(**card) for card in result['flashcards']]

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
    return {"subjects": ["history", "science", "technology", "general"], "default": "general"}

@app.get("/difficulty-levels")
async def get_difficulty_levels():
    return {"levels": ["easy", "medium", "hard"], "default": "medium"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
