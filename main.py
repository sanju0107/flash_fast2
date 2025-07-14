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

# FastAPI app
app = FastAPI(title="Advanced Flashcard API", version="2.0.0")

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

# Advanced flashcard generator (TurboLearn.ai style)
class AdvancedFlashcardGenerator:
    def __init__(self):
        self.subjects = {
            'history': ['century', 'empire', 'revolution', 'war', 'treaty', 'dynasty', 'civilization'],
            'science': ['theory', 'experiment', 'hypothesis', 'discovery', 'research', 'analysis'],
            'geography': ['continent', 'climate', 'population', 'economy', 'culture', 'region'],
            'biology': ['species', 'evolution', 'genetics', 'ecosystem', 'organism', 'cell'],
            'physics': ['force', 'energy', 'motion', 'wave', 'particle', 'quantum'],
            'chemistry': ['compound', 'reaction', 'element', 'molecule', 'bond', 'solution'],
            'math': ['theorem', 'proof', 'equation', 'function', 'derivative', 'integral'],
            'literature': ['theme', 'character', 'narrative', 'symbolism', 'metaphor', 'genre'],
            'economics': ['market', 'inflation', 'supply', 'demand', 'investment', 'trade'],
            'technology': ['algorithm', 'system', 'network', 'protocol', 'architecture', 'framework'],
            'medicine': ['diagnosis', 'treatment', 'symptom', 'disease', 'therapy', 'clinical'],
            'law': ['statute', 'jurisdiction', 'precedent', 'contract', 'liability', 'court'],
            'philosophy': ['ethics', 'logic', 'metaphysics', 'epistemology', 'consciousness', 'reasoning'],
            'environment': ['climate', 'sustainability', 'ecosystem', 'conservation', 'pollution', 'renewable']
        }
        
        # Stop words and common terms to avoid
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'this', 'that', 'these', 'those', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'
        }
        
        # Quality indicators for educational content
        self.quality_indicators = {
            'high': ['fundamental', 'essential', 'critical', 'primary', 'key', 'main', 'central', 'core'],
            'medium': ['important', 'significant', 'notable', 'relevant', 'major', 'substantial'],
            'low': ['additional', 'supplementary', 'minor', 'secondary', 'optional']
        }
    
    def detect_subject(self, content: str) -> str:
        """Enhanced subject detection with context awareness"""
        content_lower = content.lower()
        scores = defaultdict(float)
        
        # Weight keywords by frequency and context
        for subject, keywords in self.subjects.items():
            for keyword in keywords:
                # Count occurrences with context weighting
                occurrences = content_lower.count(keyword)
                if occurrences > 0:
                    # Weight by word importance and frequency
                    context_weight = min(2.0, 1.0 + (occurrences * 0.2))
                    scores[subject] += context_weight
        
        # Boost score if subject-specific patterns are found
        if 'climate' in content_lower or 'environmental' in content_lower:
            scores['environment'] += 2.0
        if any(word in content_lower for word in ['treaty', 'agreement', 'convention']):
            scores['law'] += 1.5
        
        return max(scores, key=scores.get) if scores else 'general'
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract clean, complete sentences"""
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        clean_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and len(sentence) < 500:  # Reasonable length
                # Remove leading/trailing whitespace and normalize
                sentence = ' '.join(sentence.split())
                if sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                clean_sentences.append(sentence)
        
        return clean_sentences
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract important terms and concepts"""
        # Remove punctuation and split into words
        words = re.findall(r'\b[A-Za-z]+\b', text.lower())
        
        # Filter out stop words and short words
        key_terms = []
        for word in words:
            if (len(word) > 3 and 
                word not in self.stop_words and 
                word.isalpha() and 
                not word.isdigit()):
                key_terms.append(word)
        
        # Count frequency and return most common
        term_counts = Counter(key_terms)
        return [term for term, count in term_counts.most_common(20) if count > 1]
    
    def create_definition_cards(self, sentences: List[str]) -> List[Dict]:
        """Create high-quality definition flashcards"""
        cards = []
        
        # Enhanced patterns for definitions
        definition_patterns = [
            # "X is Y" - most common definition pattern
            (r'^([A-Z][^,]+?)\s+is\s+(.+?)(?:\.|$)', 0.95),
            # "X refers to Y"
            (r'^([A-Z][^,]+?)\s+refers?\s+to\s+(.+?)(?:\.|$)', 0.90),
            # "X means Y"
            (r'^([A-Z][^,]+?)\s+means?\s+(.+?)(?:\.|$)', 0.90),
            # "X can be defined as Y"
            (r'^([A-Z][^,]+?)\s+(?:can\s+be\s+)?defined\s+as\s+(.+?)(?:\.|$)', 0.85),
            # "X represents Y"
            (r'^([A-Z][^,]+?)\s+represents?\s+(.+?)(?:\.|$)', 0.80),
            # "X involves Y"
            (r'^([A-Z][^,]+?)\s+involves?\s+(.+?)(?:\.|$)', 0.75),
        ]
        
        for sentence in sentences:
            for pattern, confidence in definition_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    term = match.group(1).strip()
                    definition = match.group(2).strip()
                    
                    # Quality checks
                    if (len(term) > 3 and len(term) < 100 and 
                        len(definition) > 10 and len(definition) < 300 and
                        term.lower() not in self.stop_words):
                        
                        cards.append({
                            'type': 'definition',
                            'question': f"What is {term}?",
                            'answer': definition.capitalize(),
                            'confidence_score': confidence,
                            'educational_value': self._calculate_educational_value(term, definition)
                        })
                        break  # Only use first matching pattern
        
        return cards
    
    def create_factual_cards(self, sentences: List[str]) -> List[Dict]:
        """Create factual Q&A cards"""
        cards = []
        
        for sentence in sentences:
            # Extract years with context
            year_matches = re.finditer(r'\b(19|20)\d{2}\b', sentence)
            for match in year_matches:
                year = match.group()
                # Get meaningful context around the year
                start = max(0, match.start() - 30)
                end = min(len(sentence), match.end() + 30)
                context = sentence[start:end].strip()
                
                # Clean context and create question
                if len(context) > 20:
                    context_clean = context.replace(year, '[YEAR]')
                    question = f"In what year {context_clean.lower()}?"
                    question = question.replace('[YEAR]', '___')
                    
                    cards.append({
                        'type': 'fact',
                        'question': question.capitalize(),
                        'answer': year,
                        'confidence_score': 0.85,
                        'educational_value': self._calculate_educational_value(year, context)
                    })
            
            # Extract percentages and numbers
            number_pattern = r'(\d+(?:\.\d+)?)\s*(%|percent|million|billion|thousand|degrees?|meters?|kilometers?|years?)'
            for match in re.finditer(number_pattern, sentence):
                number, unit = match.groups()
                context = sentence[:match.start()] + '___' + sentence[match.end():]
                
                if len(context) > 30:
                    cards.append({
                        'type': 'fact',
                        'question': f"What is the number: {context}?",
                        'answer': f"{number} {unit}",
                        'confidence_score': 0.80,
                        'educational_value': self._calculate_educational_value(f"{number} {unit}", sentence)
                    })
        
        return cards
    
    def create_conceptual_cards(self, sentences: List[str], key_terms: List[str]) -> List[Dict]:
        """Create conceptual understanding cards"""
        cards = []
        
        for sentence in sentences:
            # Look for cause-effect relationships
            cause_patterns = [
                r'(.+?)\s+(?:leads?\s+to|results?\s+in|causes?)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:due\s+to|because\s+of|as\s+a\s+result\s+of)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:triggers?|produces?|creates?)\s+(.+?)(?:\.|$)',
            ]
            
            for pattern in cause_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    cause = match.group(1).strip()
                    effect = match.group(2).strip()
                    
                    if len(cause) > 10 and len(effect) > 10:
                        cards.append({
                            'type': 'causation',
                            'question': f"What leads to {effect}?",
                            'answer': cause.capitalize(),
                            'confidence_score': 0.85,
                            'educational_value': self._calculate_educational_value(cause, effect)
                        })
                        break
            
            # Create comparison cards
            comparison_patterns = [
                r'(.+?)\s+(?:unlike|compared\s+to|in\s+contrast\s+to|different\s+from)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:while|whereas|however)\s+(.+?)(?:\.|$)',
            ]
            
            for pattern in comparison_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    item1 = match.group(1).strip()
                    item2 = match.group(2).strip()
                    
                    if len(item1) > 10 and len(item2) > 10:
                        cards.append({
                            'type': 'comparison',
                            'question': f"How does {item1} differ from {item2}?",
                            'answer': sentence,
                            'confidence_score': 0.80,
                            'educational_value': self._calculate_educational_value(item1, item2)
                        })
                        break
        
        return cards
    
    def create_application_cards(self, sentences: List[str]) -> List[Dict]:
        """Create application and example cards"""
        cards = []
        
        for sentence in sentences:
            # Look for examples
            example_patterns = [
                r'(?:for\s+example|such\s+as|including|like)\s+(.+?)(?:\.|$)',
                r'(.+?)\s+(?:is\s+an\s+example\s+of|exemplifies)\s+(.+?)(?:\.|$)',
            ]
            
            for pattern in example_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 1:
                        example = match.group(1).strip()
                        context = sentence[:match.start()].strip()
                        
                        if len(example) > 10 and len(context) > 10:
                            cards.append({
                                'type': 'example',
                                'question': f"What are examples of {context}?",
                                'answer': example.capitalize(),
                                'confidence_score': 0.75,
                                'educational_value': self._calculate_educational_value(example, context)
                            })
                    else:
                        example = match.group(1).strip()
                        concept = match.group(2).strip()
                        
                        if len(example) > 10 and len(concept) > 10:
                            cards.append({
                                'type': 'example',
                                'question': f"What is {example} an example of?",
                                'answer': concept.capitalize(),
                                'confidence_score': 0.75,
                                'educational_value': self._calculate_educational_value(example, concept)
                            })
                    break
        
        return cards
    
    def _calculate_educational_value(self, term1: str, term2: str) -> float:
        """Calculate educational value of a flashcard"""
        score = 0.5  # Base score
        
        combined_text = f"{term1} {term2}".lower()
        
        # Boost for quality indicators
        for quality_level, indicators in self.quality_indicators.items():
            for indicator in indicators:
                if indicator in combined_text:
                    if quality_level == 'high':
                        score += 0.3
                    elif quality_level == 'medium':
                        score += 0.2
                    elif quality_level == 'low':
                        score -= 0.1
        
        # Boost for technical terms
        if any(char.isupper() for char in term1) or any(char.isupper() for char in term2):
            score += 0.1
        
        # Boost for numbers and dates
        if re.search(r'\d+', combined_text):
            score += 0.1
        
        # Penalize very short or very long content
        total_length = len(term1) + len(term2)
        if total_length < 20:
            score -= 0.2
        elif total_length > 300:
            score -= 0.1
        
        return min(1.0, max(0.1, score))
    
    def rank_and_filter_cards(self, cards: List[Dict], max_cards: int) -> List[Dict]:
        """Rank cards by quality and filter duplicates"""
        # Remove duplicates based on question similarity
        unique_cards = []
        seen_questions = set()
        
        for card in cards:
            question_key = re.sub(r'\s+', ' ', card['question'].lower().strip())
            question_key = re.sub(r'[^\w\s]', '', question_key)
            
            if question_key not in seen_questions:
                seen_questions.add(question_key)
                unique_cards.append(card)
        
        # Sort by educational value and confidence
        unique_cards.sort(key=lambda x: (
            x.get('educational_value', 0.5) * 0.6 + 
            x['confidence_score'] * 0.4
        ), reverse=True)
        
        # Ensure variety in card types
        type_counts = defaultdict(int)
        final_cards = []
        
        for card in unique_cards:
            card_type = card['type']
            type_limit = {
                'definition': max_cards // 3,
                'fact': max_cards // 4,
                'causation': max_cards // 5,
                'comparison': max_cards // 6,
                'example': max_cards // 6
            }
            
            if type_counts[card_type] < type_limit.get(card_type, max_cards // 8):
                final_cards.append(card)
                type_counts[card_type] += 1
                
                if len(final_cards) >= max_cards:
                    break
        
        return final_cards
    
    def generate_flashcards(self, content: str, subject: Optional[str] = None, 
                          max_flashcards: int = 50, difficulty: str = "medium") -> Dict:
        start_time = time.time()
        
        try:
            # Subject detection
            if not subject:
                subject = self.detect_subject(content)
            
            # Extract sentences and key terms
            sentences = self.extract_sentences(content)
            key_terms = self.extract_key_terms(content)
            
            # Generate different types of cards
            all_cards = []
            
            # Definition cards (highest priority)
            definition_cards = self.create_definition_cards(sentences)
            all_cards.extend(definition_cards)
            
            # Factual cards
            factual_cards = self.create_factual_cards(sentences)
            all_cards.extend(factual_cards)
            
            # Conceptual cards
            conceptual_cards = self.create_conceptual_cards(sentences, key_terms)
            all_cards.extend(conceptual_cards)
            
            # Application cards
            application_cards = self.create_application_cards(sentences)
            all_cards.extend(application_cards)
            
            # Add subject and difficulty to all cards
            for card in all_cards:
                card['subject'] = subject
                card['difficulty'] = difficulty
            
            # Rank and filter
            final_cards = self.rank_and_filter_cards(all_cards, max_flashcards)
            
            processing_time = time.time() - start_time
            
            return {
                'flashcards': final_cards,
                'metadata': {
                    'subject': subject,
                    'sentences_processed': len(sentences),
                    'key_terms_found': len(key_terms),
                    'final_count': len(final_cards),
                    'difficulty': difficulty,
                    'extraction_stats': {
                        'definition_cards': len(definition_cards),
                        'factual_cards': len(factual_cards),
                        'conceptual_cards': len(conceptual_cards),
                        'application_cards': len(application_cards),
                        'total_generated': len(all_cards),
                        'final_selected': len(final_cards)
                    }
                },
                'performance': {
                    'processing_time': processing_time,
                    'flashcards_per_second': len(final_cards) / max(processing_time, 0.1)
                },
                'success': True,
                'message': f"Generated {len(final_cards)} high-quality educational flashcards"
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
generator = AdvancedFlashcardGenerator()

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="2.0.0"
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