import os
import re
import time
from datetime import datetime
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# FastAPI app
app = FastAPI(title="Flashcard API", version="1.0.0")

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

# Simple flashcard generator
class SimpleFlashcardGenerator:
    def __init__(self):
        self.subjects = {
            'history': ['date', 'year', 'war', 'battle', 'empire'],
            'geography': ['capital', 'river', 'mountain', 'country'],
            'science': ['process', 'element', 'reaction', 'cell'],
            'math': ['formula', 'equation', 'theorem', 'solve'],
            'literature': ['author', 'character', 'theme', 'novel']
        }
    
    def detect_subject(self, content: str) -> str:
        content_lower = content.lower()
        scores = {}
        
        for subject, keywords in self.subjects.items():
            score = sum(content_lower.count(word) for word in keywords)
            scores[subject] = score
        
        return max(scores, key=scores.get) if scores else 'general'
    
    def create_chunks(self, content: str) -> List[str]:
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > 500:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += ". " + sentence if current_chunk else sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def generate_from_chunk(self, chunk: str, subject: str, difficulty: str) -> List[Dict]:
        cards = []
        sentences = [s.strip() for s in chunk.split('.') if s.strip()]
        
        for sentence in sentences[:5]:  # Limit to prevent too many cards
            if len(sentence) < 20:
                continue
                
            # Simple definition card
            if ' is ' in sentence:
                parts = sentence.split(' is ', 1)
                if len(parts) == 2:
                    cards.append({
                        'question': f"What is {parts[0].strip()}?",
                        'answer': parts[1].strip(),
                        'type': 'definition',
                        'subject': subject,
                        'difficulty': difficulty,
                        'confidence_score': 0.7
                    })
            
            # Year/date card
            years = re.findall(r'\b(19|20)\d{2}\b', sentence)
            for year in years:
                context = sentence.replace(year, "___")
                cards.append({
                    'question': f"What year: {context}",
                    'answer': year,
                    'type': 'fact',
                    'subject': subject,
                    'difficulty': difficulty,
                    'confidence_score': 0.8
                })
            
            # General question
            if len(sentence) > 30:
                cards.append({
                    'question': f"What can you tell me about: {sentence[:50]}...?",
                    'answer': sentence,
                    'type': 'general',
                    'subject': subject,
                    'difficulty': difficulty,
                    'confidence_score': 0.6
                })
        
        return cards[:6]  # Max 6 cards per chunk
    
    def generate_flashcards(self, content: str, subject: Optional[str] = None, 
                          max_flashcards: int = 50, difficulty: str = "medium") -> Dict:
        start_time = time.time()
        
        try:
            if not subject:
                subject = self.detect_subject(content)
            
            chunks = self.create_chunks(content)
            all_cards = []
            
            for chunk in chunks:
                chunk_cards = self.generate_from_chunk(chunk, subject, difficulty)
                all_cards.extend(chunk_cards)
            
            # Remove duplicates and limit
            seen = set()
            unique_cards = []
            for card in all_cards:
                if card['question'] not in seen:
                    seen.add(card['question'])
                    unique_cards.append(card)
            
            final_cards = unique_cards[:max_flashcards]
            processing_time = time.time() - start_time
            
            return {
                'flashcards': final_cards,
                'metadata': {
                    'subject': subject,
                    'total_chunks': len(chunks),
                    'final_count': len(final_cards),
                    'difficulty': difficulty
                },
                'performance': {
                    'processing_time': processing_time,
                    'flashcards_per_second': len(final_cards) / max(processing_time, 0.1)
                },
                'success': True,
                'message': f"Generated {len(final_cards)} flashcards"
            }
            
        except Exception as e:
            return {
                'flashcards': [],
                'metadata': {},
                'performance': {},
                'success': False,
                'message': f"Error: {str(e)}"
            }

# Generator instance
generator = SimpleFlashcardGenerator()

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
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
    port = int(os.environ.get("PORT", 10000))  # Render default
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
