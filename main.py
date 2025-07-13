import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import re
import time
import asyncio
from datetime import datetime
import uvicorn
import logging
from collections import defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app instance
app = FastAPI(
    title="Flashcard Generation API",
    description="Production-ready API for generating flashcards from educational content",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class FlashcardRequest(BaseModel):
    content: str = Field(..., min_length=100, max_length=10000, description="Content to generate flashcards from")
    subject: Optional[str] = Field(None, description="Optional subject hint (auto-detected if not provided)")
    max_flashcards: Optional[int] = Field(50, ge=10, le=100, description="Maximum number of flashcards to generate")
    difficulty_level: Optional[str] = Field("medium", description="Difficulty level: easy, medium, hard")

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

# Enhanced Flashcard Generator Class
class FastAPIFlashcardGenerator:
    """
    Enhanced flashcard generator optimized for FastAPI deployment
    """
    
    def __init__(self):
        self.subject_patterns = {
            'history': {
                'keywords': ['date', 'year', 'century', 'period', 'dynasty', 'empire', 'war', 'revolution', 'treaty', 'battle', 'ancient', 'medieval'],
                'question_starters': ['When did', 'Who was', 'What was', 'Where did', 'Why did', 'How did', 'Which year'],
                'patterns': [
                    r'(\d{4})\s*[-–]\s*(\d{4})',
                    r'in\s+(\d{4})',
                    r'(Battle of \w+)',
                    r'(\w+\s+Empire)',
                    r'(\w+\s+Dynasty)',
                    r'(\w+\s+Revolution)',
                ]
            },
            'geography': {
                'keywords': ['located', 'capital', 'river', 'mountain', 'climate', 'population', 'country', 'continent', 'ocean', 'desert'],
                'question_starters': ['Where is', 'What is the capital of', 'Which river', 'What type of climate', 'How many people'],
                'patterns': [
                    r'capital of (\w+)',
                    r'(\w+\s+River)',
                    r'(\w+\s+Mountains?)',
                    r'population of ([\d,]+)',
                    r'(\w+\s+Ocean)',
                    r'(\w+\s+Desert)',
                ]
            },
            'mathematics': {
                'keywords': ['formula', 'equation', 'theorem', 'property', 'rule', 'method', 'calculate', 'solve', 'proof'],
                'question_starters': ['What is', 'How do you calculate', 'What is the formula for', 'Solve for', 'Prove that'],
                'patterns': [
                    r'([A-Z][a-z]+\'s\s+theorem)',
                    r'([A-Z][a-z]+\'s\s+rule)',
                    r'(\w+\s+formula)',
                    r'(∫|∑|∆|∂)',  # Mathematical symbols
                ]
            },
            'science': {
                'keywords': ['process', 'function', 'structure', 'element', 'compound', 'reaction', 'cell', 'organ', 'molecule', 'atom'],
                'question_starters': ['What is', 'How does', 'What happens when', 'Why does', 'Describe the'],
                'patterns': [
                    r'([A-Z][a-z]+\s+process)',
                    r'([A-Z][a-z]+\s+reaction)',
                    r'(DNA|RNA|ATP|pH)',
                    r'(\w+\s+cell)',
                ]
            },
            'literature': {
                'keywords': ['author', 'character', 'theme', 'plot', 'setting', 'metaphor', 'symbol', 'novel', 'poem'],
                'question_starters': ['Who wrote', 'What is the theme of', 'Describe the character', 'What does the symbol'],
                'patterns': [
                    r'written by (\w+)',
                    r'(\w+\s+wrote)',
                    r'(main character)',
                ]
            }
        }
        
        self.difficulty_settings = {
            'easy': {'min_words': 5, 'max_words': 15, 'complexity_threshold': 0.3},
            'medium': {'min_words': 10, 'max_words': 25, 'complexity_threshold': 0.6},
            'hard': {'min_words': 15, 'max_words': 40, 'complexity_threshold': 0.9}
        }
    
    async def detect_subject(self, content: str) -> str:
        """Async subject detection with improved accuracy"""
        word_count = len(content.split())
        subject_scores = {}
        
        for subject, patterns in self.subject_patterns.items():
            score = 0
            
            # Keyword matching
            for keyword in patterns['keywords']:
                score += content.lower().count(keyword) * 2
            
            # Pattern matching
            for pattern in patterns['patterns']:
                try:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    score += len(matches) * 3
                except re.error:
                    continue
            
            subject_scores[subject] = score / max(word_count, 1)
        
        return max(subject_scores, key=subject_scores.get) if subject_scores else 'general'
    
    async def create_smart_chunks(self, content: str) -> List[str]:
        """Create optimized chunks for processing"""
        # Split by sentences first for better context preservation
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        current_word_count = 0
        min_words, max_words = 100, 150
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_word_count + sentence_words > max_words and current_chunk:
                if current_word_count >= min_words:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_word_count = sentence_words
                else:
                    current_chunk += ". " + sentence
                    current_word_count += sentence_words
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
                current_word_count += sentence_words
        
        if current_chunk and current_word_count >= min_words:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def calculate_confidence_score(self, question: str, answer: str, subject: str) -> float:
        """Calculate confidence score for flashcard quality"""
        score = 0.5  # Base score
        
        # Length appropriateness
        if 10 <= len(question) <= 100:
            score += 0.1
        if 5 <= len(answer) <= 200:
            score += 0.1
        
        # Subject relevance
        if subject in self.subject_patterns:
            keywords = self.subject_patterns[subject]['keywords']
            relevance = sum(1 for keyword in keywords if keyword in question.lower() or keyword in answer.lower())
            score += min(relevance * 0.05, 0.2)
        
        # Question quality
        question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'which']
        if any(word in question.lower() for word in question_words):
            score += 0.1
        
        return min(score, 1.0)
    
    async def generate_flashcards_from_chunk(self, chunk: str, subject: str, difficulty: str, max_cards: int = 6) -> List[Dict]:
        """Generate flashcards from a chunk with async processing"""
        flashcards = []
        sentences = [s.strip() for s in chunk.split('.') if s.strip()]
        
        # Generate different types of flashcards
        card_generators = [
            self._generate_definition_cards,
            self._generate_fact_cards,
            self._generate_relationship_cards,
            self._generate_application_cards
        ]
        
        for generator in card_generators:
            cards = await generator(sentences, subject, difficulty)
            flashcards.extend(cards)
            
            if len(flashcards) >= max_cards:
                break
        
        # Add confidence scores
        for card in flashcards:
            card['confidence_score'] = self.calculate_confidence_score(
                card['question'], card['answer'], subject
            )
        
        return flashcards[:max_cards]
    
    async def _generate_definition_cards(self, sentences: List[str], subject: str, difficulty: str) -> List[Dict]:
        """Generate definition-based flashcards"""
        cards = []
        
        for sentence in sentences:
            # Look for definition patterns
            patterns = [
                r'(\w+(?:\s+\w+)*)\s+is\s+(.+)',
                r'(\w+(?:\s+\w+)*)\s+refers to\s+(.+)',
                r'(\w+(?:\s+\w+)*)\s+means\s+(.+)',
                r'(\w+(?:\s+\w+)*)\s+can be defined as\s+(.+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    term = match.group(1).strip()
                    definition = match.group(2).strip()
                    
                    if len(term) > 2 and len(definition) > 10:
                        cards.append({
                            'question': f"What is {term}?",
                            'answer': definition,
                            'type': 'definition',
                            'subject': subject,
                            'difficulty': difficulty
                        })
        
        return cards
    
    async def _generate_fact_cards(self, sentences: List[str], subject: str, difficulty: str) -> List[Dict]:
        """Generate fact-based flashcards"""
        cards = []
        
        for sentence in sentences:
            # Date facts
            date_pattern = r'(\d{4})'
            dates = re.findall(date_pattern, sentence)
            
            for date in dates:
                context = sentence.replace(date, "___")
                cards.append({
                    'question': f"Fill in the blank: {context}",
                    'answer': date,
                    'type': 'fact',
                    'subject': subject,
                    'difficulty': difficulty
                })
            
            # Number facts
            if subject == 'geography':
                number_pattern = r'([\d,]+)'
                numbers = re.findall(number_pattern, sentence)
                for number in numbers:
                    if ',' in number:  # Likely population or measurement
                        context = sentence.replace(number, "___")
                        cards.append({
                            'question': f"What is the number: {context}",
                            'answer': number,
                            'type': 'fact',
                            'subject': subject,
                            'difficulty': difficulty
                        })
        
        return cards
    
    async def _generate_relationship_cards(self, sentences: List[str], subject: str, difficulty: str) -> List[Dict]:
        """Generate relationship-based flashcards"""
        cards = []
        
        relationship_patterns = [
            (r'(.+)\s+caused\s+(.+)', 'What caused {}?'),
            (r'(.+)\s+led to\s+(.+)', 'What led to {}?'),
            (r'(.+)\s+resulted in\s+(.+)', 'What resulted in {}?'),
            (r'(.+)\s+because of\s+(.+)', 'Why did {} happen?'),
        ]
        
        for sentence in sentences:
            for pattern, question_template in relationship_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    cause = match.group(1).strip()
                    effect = match.group(2).strip()
                    
                    if len(cause) > 5 and len(effect) > 5:
                        cards.append({
                            'question': question_template.format(effect),
                            'answer': cause,
                            'type': 'relationship',
                            'subject': subject,
                            'difficulty': difficulty
                        })
        
        return cards
    
    async def _generate_application_cards(self, sentences: List[str], subject: str, difficulty: str) -> List[Dict]:
        """Generate application-based flashcards"""
        cards = []
        
        if subject == 'mathematics':
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['formula', 'equation', 'solve', 'calculate']):
                    cards.append({
                        'question': "How do you apply this concept?",
                        'answer': sentence,
                        'type': 'application',
                        'subject': subject,
                        'difficulty': difficulty
                    })
        
        elif subject == 'science':
            for sentence in sentences:
                if any(word in sentence.lower() for word in ['process', 'method', 'procedure']):
                    cards.append({
                        'question': "Describe the process mentioned.",
                        'answer': sentence,
                        'type': 'application',
                        'subject': subject,
                        'difficulty': difficulty
                    })
        
        return cards
    
    async def quality_filter(self, flashcards: List[Dict]) -> List[Dict]:
        """Filter flashcards based on quality metrics"""
        filtered = []
        seen_questions = set()
        
        for card in flashcards:
            # Avoid duplicates
            if card['question'].lower() in seen_questions:
                continue
            
            # Length validation
            if len(card['question']) < 5 or len(card['answer']) < 3:
                continue
            
            # Confidence threshold
            if card.get('confidence_score', 0) < 0.3:
                continue
            
            seen_questions.add(card['question'].lower())
            filtered.append(card)
        
        return filtered
    
    async def generate_flashcards(self, content: str, subject: Optional[str] = None, 
                                max_flashcards: int = 50, difficulty: str = "medium") -> Dict:
        """Main async method to generate flashcards"""
        start_time = time.time()
        
        try:
            # Detect subject if not provided
            if not subject:
                subject = await self.detect_subject(content)
            
            # Create chunks
            chunks = await self.create_smart_chunks(content)
            
            # Generate flashcards from all chunks
            all_flashcards = []
            
            for chunk in chunks:
                chunk_cards = await self.generate_flashcards_from_chunk(
                    chunk, subject, difficulty, max_cards=6
                )
                all_flashcards.extend(chunk_cards)
            
            # Quality filtering
            quality_flashcards = await self.quality_filter(all_flashcards)
            
            # Sort by confidence score and limit
            quality_flashcards.sort(key=lambda x: x.get('confidence_score', 0), reverse=True)
            final_flashcards = quality_flashcards[:max_flashcards]
            
            processing_time = time.time() - start_time
            
            return {
                'flashcards': final_flashcards,
                'metadata': {
                    'subject': subject,
                    'total_chunks': len(chunks),
                    'total_generated': len(all_flashcards),
                    'after_filtering': len(quality_flashcards),
                    'final_count': len(final_flashcards),
                    'difficulty': difficulty
                },
                'performance': {
                    'processing_time': processing_time,
                    'flashcards_per_second': len(final_flashcards) / max(processing_time, 0.1),
                    'chunks_per_second': len(chunks) / max(processing_time, 0.1)
                },
                'success': True,
                'message': f"Successfully generated {len(final_flashcards)} flashcards"
            }
            
        except Exception as e:
            logger.error(f"Error generating flashcards: {str(e)}")
            return {
                'flashcards': [],
                'metadata': {},
                'performance': {},
                'success': False,
                'message': f"Error: {str(e)}"
            }

# Global generator instance
generator = FastAPIFlashcardGenerator()

# API Routes
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/generate-flashcards", response_model=GenerationResult)
async def generate_flashcards_endpoint(request: FlashcardRequest):
    """
    Generate flashcards from educational content
    
    - **content**: The educational content to process (100-10000 characters)
    - **subject**: Optional subject hint (auto-detected if not provided)
    - **max_flashcards**: Maximum number of flashcards to generate (10-100)
    - **difficulty_level**: Difficulty level (easy, medium, hard)
    """
    try:
        # Validate content length
        if len(request.content) < 100:
            raise HTTPException(status_code=400, detail="Content must be at least 100 characters long")
        
        # Generate flashcards
        result = await generator.generate_flashcards(
            content=request.content,
            subject=request.subject,
            max_flashcards=request.max_flashcards,
            difficulty=request.difficulty_level
        )
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result['message'])
        
        # Convert to response format
        flashcard_responses = [
            FlashcardResponse(
                question=card['question'],
                answer=card['answer'],
                type=card['type'],
                subject=card['subject'],
                difficulty=card['difficulty'],
                confidence_score=card.get('confidence_score', 0.5)
            )
            for card in result['flashcards']
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
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/subjects")
async def get_supported_subjects():
    """Get list of supported subjects"""
    return {
        "subjects": list(generator.subject_patterns.keys()),
        "default": "general"
    }

@app.get("/difficulty-levels")
async def get_difficulty_levels():
    """Get available difficulty levels"""
    return {
        "levels": list(generator.difficulty_settings.keys()),
        "default": "medium"
    }

# Get port from environment variable or default to 8000
port = int(os.environ.get("PORT", 8000))
host = os.environ.get("HOST", "0.0.0.0")

# For deployment
if __name__ == "__main__":
    uvicorn.run(app, host=host, port=port)