import os
import re
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

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

# Enhanced flashcard generator
class EnhancedFlashcardGenerator:
    def __init__(self):
        self.subjects = {
            'history': ['date', 'year', 'war', 'battle', 'empire', 'dynasty', 'revolution', 'treaty'],
            'geography': ['capital', 'river', 'mountain', 'country', 'continent', 'ocean', 'climate'],
            'science': ['process', 'element', 'reaction', 'cell', 'molecule', 'theory', 'experiment'],
            'math': ['formula', 'equation', 'theorem', 'solve', 'calculate', 'proof', 'function'],
            'literature': ['author', 'character', 'theme', 'novel', 'poem', 'metaphor', 'symbolism'],
            'medicine': ['disease', 'symptom', 'treatment', 'diagnosis', 'therapy', 'anatomy'],
            'technology': ['algorithm', 'system', 'software', 'hardware', 'network', 'protocol']
        }
        
        # Pattern groups for different flashcard types
        self.patterns = {
            'definition': [
                r'(.+?)\s+is\s+(.+?)(?:\.|$)',
                r'(.+?)\s+refers\s+to\s+(.+?)(?:\.|$)',
                r'(.+?)\s+means\s+(.+?)(?:\.|$)',
                r'(.+?)\s+represents\s+(.+?)(?:\.|$)',
                r'(.+?)\s+defines?\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:can be|are)\s+defined\s+as\s+(.+?)(?:\.|$)',
            ],
            'causation': [
                r'(.+?)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:due\s+to|because\s+of|as\s+a\s+result\s+of)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:triggers?|produces?|creates?)\s+(.+?)(?:\.|$)',
            ],
            'comparison': [
                r'(.+?)\s+(?:unlike|compared\s+to|in\s+contrast\s+to)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:while|whereas)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:similar\s+to|like)\s+(.+?)(?:\.|$)',
            ],
            'process': [
                r'(.+?)\s+(?:involves?|includes?|consists?\s+of)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:begins?|starts?)\s+(?:with|by)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:follows?|occurs?)\s+(?:after|before)\s+(.+?)(?:\.|$)',
            ],
            'classification': [
                r'(.+?)\s+(?:types?|kinds?|categories?)\s+(?:of|include)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:classified|categorized|grouped)\s+(?:as|into)\s+(.+?)(?:\.|$)',
            ]
        }
    
    def detect_subject(self, content: str) -> str:
        content_lower = content.lower()
        scores = {}
        
        for subject, keywords in self.subjects.items():
            score = sum(content_lower.count(word) for word in keywords)
            scores[subject] = score
        
        return max(scores, key=scores.get) if any(scores.values()) else 'general'
    
    def extract_facts(self, text: str) -> List[Dict]:
        """Extract factual information like dates, numbers, names, places"""
        facts = []
        
        # Extract years and dates
        year_pattern = r'\b(?:in\s+)?(\d{4})\b'
        date_pattern = r'\b(?:on\s+)?(\w+\s+\d{1,2},\s+\d{4})\b'
        
        for match in re.finditer(year_pattern, text):
            year = match.group(1)
            context = text[max(0, match.start()-50):match.end()+50].strip()
            if len(context) > 20:
                facts.append({
                    'type': 'fact',
                    'question': f"In what year did the following occur: {context.replace(year, '___')}?",
                    'answer': year,
                    'confidence': 0.85
                })
        
        # Extract proper nouns (names, places)
        proper_noun_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        for match in re.finditer(proper_noun_pattern, text):
            name = match.group()
            if len(name) > 2 and name not in ['The', 'This', 'That', 'These', 'Those']:
                context = text[max(0, match.start()-30):match.end()+30].strip()
                if len(context) > 20:
                    facts.append({
                        'type': 'fact',
                        'question': f"Who or what is being referred to: {context.replace(name, '___')}?",
                        'answer': name,
                        'confidence': 0.75
                    })
        
        # Extract numbers with units
        number_pattern = r'\b(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\b'
        for match in re.finditer(number_pattern, text):
            number, unit = match.groups()
            context = text[max(0, match.start()-40):match.end()+40].strip()
            if len(context) > 20:
                facts.append({
                    'type': 'fact',
                    'question': f"What is the measurement: {context.replace(f'{number} {unit}', '___')}?",
                    'answer': f"{number} {unit}",
                    'confidence': 0.8
                })
        
        return facts
    
    def extract_pattern_based_cards(self, text: str) -> List[Dict]:
        """Extract flashcards based on linguistic patterns"""
        cards = []
        
        for card_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        term1, term2 = groups[0].strip(), groups[1].strip()
                        
                        # Skip if terms are too short or too long
                        if len(term1) < 3 or len(term2) < 3 or len(term1) > 100 or len(term2) > 200:
                            continue
                        
                        card = self._create_card_from_pattern(card_type, term1, term2)
                        if card:
                            cards.append(card)
        
        return cards
    
    def _create_card_from_pattern(self, card_type: str, term1: str, term2: str) -> Optional[Dict]:
        """Create a flashcard from extracted pattern"""
        confidence_scores = {
            'definition': 0.9,
            'causation': 0.85,
            'comparison': 0.8,
            'process': 0.8,
            'classification': 0.75
        }
        
        if card_type == 'definition':
            return {
                'type': 'definition',
                'question': f"What is {term1}?",
                'answer': term2,
                'confidence': confidence_scores[card_type]
            }
        elif card_type == 'causation':
            return {
                'type': 'causation',
                'question': f"What causes {term2}?",
                'answer': term1,
                'confidence': confidence_scores[card_type]
            }
        elif card_type == 'comparison':
            return {
                'type': 'comparison',
                'question': f"How does {term1} compare to {term2}?",
                'answer': f"{term1} is different from {term2}",
                'confidence': confidence_scores[card_type]
            }
        elif card_type == 'process':
            return {
                'type': 'process',
                'question': f"What does {term1} involve?",
                'answer': term2,
                'confidence': confidence_scores[card_type]
            }
        elif card_type == 'classification':
            return {
                'type': 'classification',
                'question': f"What are the types of {term1}?",
                'answer': term2,
                'confidence': confidence_scores[card_type]
            }
        
        return None
    
    def extract_key_concepts(self, text: str) -> List[Dict]:
        """Extract key concepts and create conceptual questions"""
        concepts = []
        
        # Find sentences with key indicator words
        key_indicators = [
            'important', 'significant', 'crucial', 'essential', 'fundamental',
            'primary', 'main', 'key', 'major', 'critical', 'notable'
        ]
        
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        for sentence in sentences:
            if len(sentence) < 30 or len(sentence) > 300:
                continue
                
            # Check for key indicators
            if any(indicator in sentence.lower() for indicator in key_indicators):
                # Extract the main concept
                words = sentence.split()
                if len(words) > 10:
                    concept = ' '.join(words[:8]) + '...'
                    concepts.append({
                        'type': 'concept',
                        'question': f"What is significant about: {concept}?",
                        'answer': sentence,
                        'confidence': 0.7
                    })
        
        return concepts
    
    def create_chunks(self, content: str) -> List[str]:
        """Create semantic chunks from content"""
        # Split by paragraphs first, then by sentences if needed
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        for paragraph in paragraphs:
            if len(paragraph) <= 800:
                chunks.append(paragraph)
            else:
                # Split long paragraphs into sentences
                sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > 800:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
                    else:
                        current_chunk += ". " + sentence if current_chunk else sentence
                
                if current_chunk:
                    chunks.append(current_chunk)
        
        return chunks
    
    def deduplicate_and_rank(self, cards: List[Dict]) -> List[Dict]:
        """Remove duplicates and rank by relevance and confidence"""
        # Remove duplicates based on question similarity
        unique_cards = []
        seen_questions = set()
        
        for card in cards:
            question_key = card['question'].lower().strip()
            if question_key not in seen_questions:
                seen_questions.add(question_key)
                unique_cards.append(card)
        
        # Sort by confidence score and type priority
        type_priority = {
            'definition': 1,
            'fact': 2,
            'causation': 3,
            'process': 4,
            'comparison': 5,
            'classification': 6,
            'concept': 7
        }
        
        unique_cards.sort(key=lambda x: (type_priority.get(x['type'], 8), -x['confidence']))
        
        return unique_cards
    
    def generate_flashcards(self, content: str, subject: Optional[str] = None, 
                          max_flashcards: int = 50, difficulty: str = "medium") -> Dict:
        start_time = time.time()
        
        try:
            if not subject:
                subject = self.detect_subject(content)
            
            chunks = self.create_chunks(content)
            all_cards = []
            
            for chunk in chunks:
                # Extract different types of flashcards
                facts = self.extract_facts(chunk)
                pattern_cards = self.extract_pattern_based_cards(chunk)
                concepts = self.extract_key_concepts(chunk)
                
                # Combine all cards
                chunk_cards = facts + pattern_cards + concepts
                
                # Add subject and difficulty to each card
                for card in chunk_cards:
                    card['subject'] = subject
                    card['difficulty'] = difficulty
                
                all_cards.extend(chunk_cards)
            
            # Deduplicate and rank
            unique_cards = self.deduplicate_and_rank(all_cards)
            
            # Limit to max_flashcards
            final_cards = unique_cards[:max_flashcards]
            processing_time = time.time() - start_time
            
            return {
                'flashcards': final_cards,
                'metadata': {
                    'subject': subject,
                    'total_chunks': len(chunks),
                    'final_count': len(final_cards),
                    'difficulty': difficulty,
                    'extraction_stats': {
                        'total_extracted': len(all_cards),
                        'after_deduplication': len(unique_cards),
                        'final_selected': len(final_cards)
                    }
                },
                'performance': {
                    'processing_time': processing_time,
                    'flashcards_per_second': len(final_cards) / max(processing_time, 0.1)
                },
                'success': True,
                'message': f"Generated {len(final_cards)} high-quality flashcards"
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
generator = EnhancedFlashcardGenerator()

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