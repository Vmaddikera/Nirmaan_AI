

import re
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import log_info, log_error
from utils.exceptions import ModelError


class RubricTranscriptAnalyzer:
    def __init__(self):
        """Initialize the analyzer with models and rubric"""
        # Load sentence transformer for semantic similarity
        try:
            print("Loading sentence transformer model...")
            self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
            print(" Sentence transformer loaded")
            log_info("Sentence transformer model loaded")
        except Exception as e:
            print(f" Warning: Could not load sentence transformer: {e}")
            log_error("Failed to load sentence transformer", e)
            self.semantic_model = None
        
        # Initialize language tool for grammar checking
        try:
            print("Loading language tool...")
            self.language_tool = language_tool_python.LanguageTool('en-US')
            print("Language tool loaded")
            log_info("Language tool loaded")
        except Exception as e:
            print(f" Warning: Could not load language tool: {e}")
            log_error("Failed to load language tool (optional)", e)
            self.language_tool = None
        
        # Initialize sentiment analyzer
        try:
            print("Loading sentiment analyzer...")
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            print(" Sentiment analyzer loaded")
            log_info("Sentiment analyzer loaded")
        except Exception as e:
            print(f" Warning: Could not load sentiment analyzer: {e}")
            log_error("Failed to load sentiment analyzer", e)
            self.sentiment_analyzer = None
        
        # Define the rubric based on Excel file
        self.rubric = self._load_rubric()
        print(f" Rubric loaded with {len(self.rubric)} criteria\n")
        log_info(f"Rubric loaded with {len(self.rubric)} criteria")
    
    def _load_rubric(self) -> List[Dict]:
        """Load rubric criteria from the Excel-based structure"""
        return [
            {
                'category': 'Content & Structure',
                'criterion': 'Salutation Level',
                'description': 'Type and quality of greeting/salutation',
                'keywords': {
                    'excellent': ['excited to introduce', 'feeling great', 'delighted', 'honored', 'pleased to meet'],
                    'good': ['good morning', 'good afternoon', 'good evening', 'good day', 'hello everyone', 'greetings'],
                    'normal': ['hi', 'hello'],
                    'none': []
                },
                'weight': 5,
                'max_score': 5,
                'scoring': {
                    'excellent': 5,
                    'good': 4,
                    'normal': 2,
                    'none': 0
                }
            },
            {
                'category': 'Content & Structure',
                'criterion': 'Keyword Presence',
                'description': 'Presence of essential personal information elements',
                'keywords': {
                    'must_have': {
                        'name': ['name', 'my name is', 'i am', "i'm", 'myself', 'call me'],
                        'age': ['years old', 'age', 'year old'],
                        'school_class': ['class', 'grade', 'school', 'studying', 'student'],
                        'family': ['family', 'parents', 'mother', 'father', 'siblings', 'brother', 'sister'],
                        'hobbies': ['hobby', 'hobbies', 'interest', 'interests', 'like to', 'enjoy', 'love to', 'passion']
                    },
                    'good_to_have': {
                        'about_family': ['special thing about', 'family is', 'they are'],
                        'location': ['from', 'live in', 'belong to'],
                        'goals': ['dream', 'goal', 'ambition', 'aspiration', 'want to', 'plan to', 'hope to'],
                        'unique': ['fun fact', 'unique', 'special', 'interesting', 'dont know about me', "don't know about me"],
                        'strengths': ['good at', 'strength', 'achievement', 'accomplished', 'proud of']
                    }
                },
                'weight': 30,
                'max_score': 30,
                'scoring': {
                    'must_have_each': 4,  # 5 items × 4 = 20 points
                    'good_to_have_each': 2  # 5 items × 2 = 10 points
                }
            },
            {
                'category': 'Content & Structure',
                'criterion': 'Flow',
                'description': 'Logical order of introduction: Salutation → Name → Mandatory details → Optional Details → Closing',
                'keywords': {
                    'salutation': ['hi', 'hello', 'good morning', 'good afternoon', 'good evening', 'greetings', 'hello everyone'],
                    'name': ['name', 'my name is', 'i am', "i'm", 'myself', 'call me'],
                    'basic_details': ['age', 'years old', 'class', 'school', 'studying', 'student', 'family'],
                    'additional': ['hobby', 'interest', 'like', 'enjoy', 'goal', 'dream', 'fun fact'],
                    'closing': ['thank you', 'thanks', 'thank you for listening', 'that\'s all', 'thats all']
                },
                'weight': 5,
                'max_score': 5,
                'scoring': {
                    'order_followed': 5,
                    'order_not_followed': 0
                }
            },
            {
                'category': 'Speech Rate',
                'criterion': 'Words Per Minute',
                'description': 'Speaking pace based on word count and duration',
                'weight': 10,
                'max_score': 10,
                'scoring': {
                    'ideal': (111, 140, 10),       # (min_wpm, max_wpm, score)
                    'fast': (141, 160, 6),
                    'too_fast': (161, 999, 2),
                    'slow': (81, 110, 6),
                    'too_slow': (0, 80, 2)
                }
            },
            {
                'category': 'Language & Grammar',
                'criterion': 'Grammar Errors',
                'description': 'Grammar correctness using LanguageTool',
                'weight': 10,
                'max_score': 10,
                'scoring': {
                    # Grammar Score = 1 - min(errors_per_100_words / 10, 1)
                    # >0.9: 10 points, 0.7-0.89: 8 points, 0.5-0.69: 6 points, 0.3-0.49: 4 points, <0.3: 2 points
                }
            },
            {
                'category': 'Language & Grammar',
                'criterion': 'Vocabulary Richness',
                'description': 'Type-Token Ratio (TTR) = Distinct words ÷ Total words',
                'weight': 10,
                'max_score': 10,
                'scoring': {
                    # 0.9-1.0: 10 points, 0.7-0.89: 8 points, 0.5-0.69: 6 points, 0.3-0.49: 4 points, 0-0.29: 2 points
                }
            },
            {
                'category': 'Clarity',
                'criterion': 'Filler Word Rate',
                'description': 'Frequency of filler words (um, uh, like, you know, so, actually, basically, right, i mean, well, kinda, sort of, okay, hmm, ah)',
                'keywords': ['um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'right', 'i mean', 'well', 'kinda', 'sort of', 'okay', 'hmm', 'ah'],
                'weight': 15,
                'max_score': 15,
                'scoring': {
                    # Filler Word Rate = (Number of filler words ÷ Total words) × 100
                    # 0-3: 15 points, 4-6: 12 points, 7-9: 9 points, 10-12: 6 points, 13+: 3 points
                }
            },
            {
                'category': 'Engagement',
                'criterion': 'Sentiment/Positivity',
                'description': 'Emotional tone and positivity using sentiment analysis',
                'weight': 15,
                'max_score': 15,
                'scoring': {
                    # >=0.9: 15 points, 0.7-0.89: 12 points, 0.5-0.69: 9 points, 0.3-0.49: 6 points, <0.3: 3 points
                }
            }
        ]
    
    def analyze(self, transcript_text: str, duration_seconds: int = None) -> Dict:
        """
        Analyze transcript using rule-based, NLP-based, and rubric-driven approaches
        
        Args:
            transcript_text: The transcript to analyze
            duration_seconds: Optional duration in seconds for WPM calculation
            
        Returns:
            Dictionary with overall score, per-criterion scores, and feedback
        """
        try:
            # Calculate basic text statistics
            words = self._get_words(transcript_text)
            word_count = len(words)
            sentence_count = len(re.split(r'[.!?]+', transcript_text.strip()))
            sentence_count = max(1, sentence_count)  # Avoid division by zero
            
            # Analyze each criterion
            criterion_results = []
            total_score = 0
            total_possible = 100
            
            for criterion in self.rubric:
                criterion_name = criterion['criterion']
                
                if criterion_name == 'Salutation Level':
                    result = self._evaluate_salutation(transcript_text, criterion)
                elif criterion_name == 'Keyword Presence':
                    result = self._evaluate_keywords(transcript_text, criterion)
                elif criterion_name == 'Flow':
                    result = self._evaluate_flow(transcript_text, criterion)
                elif criterion_name == 'Words Per Minute':
                    result = self._evaluate_speech_rate(word_count, duration_seconds, criterion)
                elif criterion_name == 'Grammar Errors':
                    result = self._evaluate_grammar(transcript_text, word_count, criterion)
                elif criterion_name == 'Vocabulary Richness':
                    result = self._evaluate_vocabulary(words, criterion)
                elif criterion_name == 'Filler Word Rate':
                    result = self._evaluate_filler_words(transcript_text, word_count, criterion)
                elif criterion_name == 'Sentiment/Positivity':
                    result = self._evaluate_sentiment(transcript_text, criterion)
                else:
                    result = {
                        'score': 0,
                        'max_score': criterion['max_score'],
                        'feedback': 'Not implemented'
                    }
                
                # Add criterion info to result
                result['criterion'] = criterion_name
                result['category'] = criterion['category']
                result['weight'] = criterion['weight']
                
                criterion_results.append(result)
                total_score += result['score']
            
            # Generate overall feedback
            overall_feedback = self._generate_overall_feedback(criterion_results, total_score)
            
            return {
                'overall_score': round(total_score, 2),
                'max_possible_score': total_possible,
                'percentage': round((total_score / total_possible) * 100, 2),
                'words': word_count,
                'sentences': sentence_count,
                'criteria': criterion_results,
                'overall_feedback': overall_feedback,
                'performance_level': self._get_performance_level(total_score)
            }
            
        except Exception as e:
            return {
                'error': f'Analysis failed: {str(e)}',
                'overall_score': 0,
                'criteria': []
            }
    
    def _evaluate_salutation(self, text: str, criterion: Dict) -> Dict:
        """Evaluate salutation level using rule-based keyword matching"""
        text_lower = text.lower()
        keywords = criterion['keywords']
        scoring = criterion['scoring']
        
        # Check for excellent salutation
        for keyword in keywords['excellent']:
            if keyword in text_lower:
                return {
                    'score': scoring['excellent'],
                    'max_score': criterion['max_score'],
                    'feedback': f'Excellent salutation detected: "{keyword}"',
                    'keywords_found': [keyword],
                    'rule_based_score': scoring['excellent'],
                    'semantic_score': None
                }
        
        # Check for good salutation
        for keyword in keywords['good']:
            if keyword in text_lower:
                return {
                    'score': scoring['good'],
                    'max_score': criterion['max_score'],
                    'feedback': f'Good salutation detected: "{keyword}"',
                    'keywords_found': [keyword],
                    'rule_based_score': scoring['good'],
                    'semantic_score': None
                }
        
        # Check for normal salutation
        for keyword in keywords['normal']:
            if keyword in text_lower:
                return {
                    'score': scoring['normal'],
                    'max_score': criterion['max_score'],
                    'feedback': f'Basic salutation detected: "{keyword}"',
                    'keywords_found': [keyword],
                    'rule_based_score': scoring['normal'],
                    'semantic_score': None
                }
        
        # No salutation found
        return {
            'score': scoring['none'],
            'max_score': criterion['max_score'],
            'feedback': 'No salutation detected. Consider starting with a greeting.',
            'keywords_found': [],
            'rule_based_score': scoring['none'],
            'semantic_score': None
        }
    
    def _evaluate_keywords(self, text: str, criterion: Dict) -> Dict:
        """Evaluate keyword presence using rule-based and semantic similarity"""
        text_lower = text.lower()
        keywords = criterion['keywords']
        scoring = criterion['scoring']
        
        score = 0
        keywords_found = []
        feedback_parts = []
        
        # Check must-have keywords (4 points each)
        must_have_found = {}
        for category, keyword_list in keywords['must_have'].items():
            found = False
            matched_keyword = None
            for keyword in keyword_list:
                if keyword in text_lower:
                    found = True
                    matched_keyword = keyword
                    break
            
            must_have_found[category] = found
            if found:
                score += scoring['must_have_each']
                keywords_found.append(f"{category}: {matched_keyword}")
        
        # Check good-to-have keywords (2 points each)
        good_to_have_found = {}
        for category, keyword_list in keywords['good_to_have'].items():
            found = False
            matched_keyword = None
            for keyword in keyword_list:
                if keyword in text_lower:
                    found = True
                    matched_keyword = keyword
                    break
            
            good_to_have_found[category] = found
            if found:
                score += scoring['good_to_have_each']
                keywords_found.append(f"{category}: {matched_keyword}")
        
        # Generate feedback
        missing_must = [k for k, v in must_have_found.items() if not v]
        missing_good = [k for k, v in good_to_have_found.items() if not v]
        
        if missing_must:
            feedback_parts.append(f"Missing essential elements: {', '.join(missing_must)}")
        else:
            feedback_parts.append("All essential elements present")
        
        if missing_good:
            feedback_parts.append(f"Could include: {', '.join(missing_good)}")
        
        # Semantic similarity check for name detection (bonus)
        semantic_score = None
        if self.semantic_model:
            name_sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
            if name_sentences:
                try:
                    reference = "My name is"
                    ref_embedding = self.semantic_model.encode([reference])
                    sent_embeddings = self.semantic_model.encode(name_sentences[:3])  # Check first 3 sentences
                    similarities = cosine_similarity(ref_embedding, sent_embeddings)[0]
                    semantic_score = float(np.max(similarities))
                except Exception as e:
                    semantic_score = None
        
        return {
            'score': min(score, criterion['max_score']),
            'max_score': criterion['max_score'],
            'feedback': '. '.join(feedback_parts),
            'keywords_found': keywords_found,
            'must_have_found': must_have_found,
            'good_to_have_found': good_to_have_found,
            'rule_based_score': score,
            'semantic_score': semantic_score
        }
    
    def _evaluate_flow(self, text: str, criterion: Dict) -> Dict:
        """Evaluate flow/order of introduction"""
        text_lower = text.lower()
        keywords = criterion['keywords']
        
        # Find positions of key elements
        positions = {}
        for category, keyword_list in keywords.items():
            for keyword in keyword_list:
                pos = text_lower.find(keyword)
                if pos != -1:
                    if category not in positions or pos < positions[category]:
                        positions[category] = pos
        
        # Check if order is correct: salutation < name < basic_details < additional < closing
        expected_order = ['salutation', 'name', 'basic_details', 'additional', 'closing']
        found_order = sorted([(cat, pos) for cat, pos in positions.items()], key=lambda x: x[1])
        found_categories = [cat for cat, _ in found_order]
        
        # Check if found categories maintain the expected order
        order_correct = True
        last_expected_idx = -1
        for cat in found_categories:
            if cat in expected_order:
                idx = expected_order.index(cat)
                if idx < last_expected_idx:
                    order_correct = False
                    break
                last_expected_idx = idx
        
        if order_correct and len(found_categories) >= 3:
            score = criterion['scoring']['order_followed']
            feedback = f"Good flow: {' → '.join(found_categories)}"
        else:
            score = criterion['scoring']['order_not_followed']
            feedback = f"Flow could be improved. Found order: {' → '.join(found_categories)}"
        
        return {
            'score': score,
            'max_score': criterion['max_score'],
            'feedback': feedback,
            'found_order': found_categories,
            'positions': positions,
            'rule_based_score': score,
            'semantic_score': None
        }
    
    def _evaluate_speech_rate(self, word_count: int, duration_seconds: int, criterion: Dict) -> Dict:
        """Evaluate speech rate (words per minute)"""
        if duration_seconds is None or duration_seconds <= 0:
            return {
                'score': criterion['max_score'] // 2,  # Give half points if duration not provided
                'max_score': criterion['max_score'],
                'feedback': 'Duration not provided. Assuming moderate pace.',
                'wpm': None,
                'rule_based_score': criterion['max_score'] // 2,
                'semantic_score': None
            }
        
        wpm = (word_count / duration_seconds) * 60
        scoring = criterion['scoring']
        
        # Determine score based on WPM
        for category, (min_wpm, max_wpm, points) in scoring.items():
            if min_wpm <= wpm <= max_wpm:
                feedback = f"Speech rate: {wpm:.0f} WPM ({category.replace('_', ' ').title()})"
                return {
                    'score': points,
                    'max_score': criterion['max_score'],
                    'feedback': feedback,
                    'wpm': round(wpm, 2),
                    'category': category,
                    'rule_based_score': points,
                    'semantic_score': None
                }
        
        # Default (shouldn't reach here)
        return {
            'score': 5,
            'max_score': criterion['max_score'],
            'feedback': f"Speech rate: {wpm:.0f} WPM",
            'wpm': round(wpm, 2),
            'rule_based_score': 5,
            'semantic_score': None
        }
    
    def _evaluate_grammar(self, text: str, word_count: int, criterion: Dict) -> Dict:
        """Evaluate grammar using LanguageTool"""
        if not self.language_tool or word_count == 0:
            # Fallback: give average score if tool not available
            return {
                'score': criterion['max_score'] * 0.7,
                'max_score': criterion['max_score'],
                'feedback': 'Grammar checking not available. Assuming good grammar.',
                'error_count': 0,
                'errors_per_100_words': 0,
                'rule_based_score': criterion['max_score'] * 0.7,
                'semantic_score': None
            }
        
        try:
            matches = self.language_tool.check(text)
            error_count = len(matches)
            errors_per_100_words = (error_count / word_count) * 100
            
            # Grammar Score = 1 - min(errors_per_100_words / 10, 1)
            grammar_ratio = max(0, 1 - min(errors_per_100_words / 10, 1))
            
            # Map to points
            if grammar_ratio > 0.9:
                score = 10
            elif grammar_ratio >= 0.7:
                score = 8
            elif grammar_ratio >= 0.5:
                score = 6
            elif grammar_ratio >= 0.3:
                score = 4
            else:
                score = 2
            
            feedback = f"{error_count} grammar errors found ({errors_per_100_words:.1f} per 100 words)"
            
            return {
                'score': score,
                'max_score': criterion['max_score'],
                'feedback': feedback,
                'error_count': error_count,
                'errors_per_100_words': round(errors_per_100_words, 2),
                'grammar_ratio': round(grammar_ratio, 3),
                'rule_based_score': score,
                'semantic_score': None
            }
        except Exception as e:
            return {
                'score': criterion['max_score'] * 0.7,
                'max_score': criterion['max_score'],
                'feedback': f'Grammar check error: {str(e)}',
                'error_count': 0,
                'rule_based_score': criterion['max_score'] * 0.7,
                'semantic_score': None
            }
    
    def _evaluate_vocabulary(self, words: List[str], criterion: Dict) -> Dict:
        """Evaluate vocabulary richness using Type-Token Ratio (TTR)"""
        if not words:
            return {
                'score': 0,
                'max_score': criterion['max_score'],
                'feedback': 'No words to analyze',
                'ttr': 0,
                'rule_based_score': 0,
                'semantic_score': None
            }
        
        # Calculate TTR = Distinct words / Total words
        distinct_words = len(set(w.lower() for w in words))
        total_words = len(words)
        ttr = distinct_words / total_words if total_words > 0 else 0
        
        # Map to points
        if ttr >= 0.9:
            score = 10
        elif ttr >= 0.7:
            score = 8
        elif ttr >= 0.5:
            score = 6
        elif ttr >= 0.3:
            score = 4
        else:
            score = 2
        
        feedback = f"Vocabulary richness (TTR): {ttr:.3f} ({distinct_words} distinct / {total_words} total words)"
        
        return {
            'score': score,
            'max_score': criterion['max_score'],
            'feedback': feedback,
            'ttr': round(ttr, 3),
            'distinct_words': distinct_words,
            'total_words': total_words,
            'rule_based_score': score,
            'semantic_score': None
        }
    
    def _evaluate_filler_words(self, text: str, word_count: int, criterion: Dict) -> Dict:
        """Evaluate filler word rate"""
        if word_count == 0:
            return {
                'score': 0,
                'max_score': criterion['max_score'],
                'feedback': 'No words to analyze',
                'filler_count': 0,
                'filler_rate': 0,
                'rule_based_score': 0,
                'semantic_score': None
            }
        
        text_lower = text.lower()
        filler_words = criterion['keywords']
        
        # Count filler words
        filler_count = 0
        found_fillers = []
        for filler in filler_words:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                filler_count += len(matches)
                found_fillers.append(f"{filler} ({len(matches)})")
        
        # Filler Word Rate = (Number of filler words / Total words) × 100
        filler_rate = (filler_count / word_count) * 100
        
        # Map to points
        if filler_rate <= 3:
            score = 15
        elif filler_rate <= 6:
            score = 12
        elif filler_rate <= 9:
            score = 9
        elif filler_rate <= 12:
            score = 6
        else:
            score = 3
        
        if filler_count == 0:
            feedback = "Excellent: No filler words detected"
        else:
            feedback = f"{filler_count} filler words found ({filler_rate:.1f}%): {', '.join(found_fillers[:5])}"
        
        return {
            'score': score,
            'max_score': criterion['max_score'],
            'feedback': feedback,
            'filler_count': filler_count,
            'filler_rate': round(filler_rate, 2),
            'found_fillers': found_fillers,
            'rule_based_score': score,
            'semantic_score': None
        }
    
    def _evaluate_sentiment(self, text: str, criterion: Dict) -> Dict:
        """Evaluate sentiment/positivity using VADER"""
        if not self.sentiment_analyzer:
            return {
                'score': criterion['max_score'] * 0.6,
                'max_score': criterion['max_score'],
                'feedback': 'Sentiment analysis not available. Assuming neutral tone.',
                'sentiment_score': 0.5,
                'rule_based_score': criterion['max_score'] * 0.6,
                'semantic_score': None
            }
        
        try:
            # Get sentiment scores
            scores = self.sentiment_analyzer.polarity_scores(text)
            positive_score = scores['pos']  # 0 to 1
            
            # Map to points
            if positive_score >= 0.9:
                score = 15
            elif positive_score >= 0.7:
                score = 12
            elif positive_score >= 0.5:
                score = 9
            elif positive_score >= 0.3:
                score = 6
            else:
                score = 3
            
            # Determine sentiment label
            if positive_score >= 0.7:
                sentiment_label = "Highly Positive"
            elif positive_score >= 0.5:
                sentiment_label = "Positive"
            elif positive_score >= 0.3:
                sentiment_label = "Neutral"
            else:
                sentiment_label = "Needs Improvement"
            
            feedback = f"Sentiment: {sentiment_label} (positivity score: {positive_score:.3f})"
            
            return {
                'score': score,
                'max_score': criterion['max_score'],
                'feedback': feedback,
                'sentiment_score': round(positive_score, 3),
                'sentiment_label': sentiment_label,
                'full_scores': scores,
                'rule_based_score': score,
                'semantic_score': None
            }
        except Exception as e:
            return {
                'score': criterion['max_score'] * 0.6,
                'max_score': criterion['max_score'],
                'feedback': f'Sentiment analysis error: {str(e)}',
                'sentiment_score': 0.5,
                'rule_based_score': criterion['max_score'] * 0.6,
                'semantic_score': None
            }
    
    def _get_words(self, text: str) -> List[str]:
        """Extract words from text"""
        # Remove punctuation and split
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        return words
    
    def _generate_overall_feedback(self, criteria: List[Dict], total_score: float) -> str:
        """Generate overall feedback based on criterion results"""
        feedback_parts = []
        
        # Identify strengths
        strengths = [c for c in criteria if c['score'] >= c['max_score'] * 0.8]
        if strengths:
            strength_names = [c['criterion'] for c in strengths]
            feedback_parts.append(f"Strengths: {', '.join(strength_names)}")
        
        # Identify areas for improvement
        weaknesses = [c for c in criteria if c['score'] < c['max_score'] * 0.5]
        if weaknesses:
            weakness_names = [c['criterion'] for c in weaknesses]
            feedback_parts.append(f"Areas for improvement: {', '.join(weakness_names)}")
        
        # Overall performance
        if total_score >= 85:
            feedback_parts.append("Excellent overall performance!")
        elif total_score >= 70:
            feedback_parts.append("Good performance with room for improvement.")
        elif total_score >= 50:
            feedback_parts.append("Moderate performance. Focus on key areas.")
        else:
            feedback_parts.append("Needs significant improvement across multiple criteria.")
        
        return ' '.join(feedback_parts)
    
    def _get_performance_level(self, score: float) -> str:
        """Get performance level based on score"""
        if score >= 85:
            return "Excellent"
        elif score >= 70:
            return "Good"
        elif score >= 50:
            return "Average"
        else:
            return "Needs Improvement"

