import pandas as pd
import numpy as np
import re
from scipy import stats
from scipy.stats import shapiro, normaltest
from tabulate import tabulate
import json
from typing import Dict, List, Tuple

class TranscriptAnalyzer:
    def __init__(self):
        self.quality_indicators = {
            'professional_language': [
                r'\b(?:implemented|developed|managed|coordinated|led|achieved|delivered|optimized|streamlined)\b',
                r'\b(?:collaborated|facilitated|mentored|supervised|trained|guided)\b',
                r'\b(?:analyzed|evaluated|assessed|reviewed|monitored|tracked)\b'
            ],
            'confidence_indicators': [
                r'\b(?:successfully|effectively|efficiently|significantly|substantially)\b',
                r'\b(?:expertise|proficiency|mastery|advanced|specialized)\b',
                r'\b(?:demonstrated|proven|established|validated)\b'
            ],
            'technical_indicators': [
                r'\b(?:algorithm|framework|methodology|architecture|infrastructure)\b',
                r'\b(?:optimization|scalability|performance|efficiency|reliability)\b',
                r'\b(?:integration|deployment|automation|testing|debugging)\b'
            ]
        }
    
    def analyze(self, transcript_text: str) -> Dict:
        """
        Analyze transcript text for quality, structure, and content
        """
        try:
            # Basic text statistics
            text_stats = self._calculate_text_statistics(transcript_text)
            
            # Quality indicators analysis
            quality_analysis = self._analyze_quality_indicators(transcript_text)
            
            # Structure analysis
            structure_analysis = self._analyze_structure(transcript_text)
            
            # Content coherence analysis
            coherence_analysis = self._analyze_coherence(transcript_text)
            
            # Professional language analysis
            professional_analysis = self._analyze_professional_language(transcript_text)
            
            # Generate overall assessment
            overall_assessment = self._generate_overall_assessment(
                text_stats, quality_analysis, structure_analysis, coherence_analysis, professional_analysis
            )
            
            # Calculate communication and clarity scores
            communication_score = overall_assessment.get('overall_score', 0.5)
            clarity_score = min(1.0, max(0.0, text_stats.get('avg_sentence_length', 15) / 20.0))
            response_quality = "Strong" if communication_score > 0.7 else "Needs Improvement"
            
            return {
                'communication_score': communication_score,
                'clarity_score': clarity_score,
                'response_quality': response_quality,
                'text_statistics': text_stats,
                'quality_indicators': quality_analysis,
                'structure_analysis': structure_analysis,
                'coherence_analysis': coherence_analysis,
                'professional_analysis': professional_analysis,
                'overall_assessment': overall_assessment,
                'summary': self._generate_transcript_summary(overall_assessment, quality_analysis)
            }
            
        except Exception as e:
            return {'error': f'Transcript analysis failed: {str(e)}'}
    
    def _calculate_text_statistics(self, text: str) -> Dict:
        """Calculate basic text statistics"""
        try:
            # Word and sentence counts
            words = text.split()
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Character counts
            char_count = len(text)
            char_count_no_spaces = len(text.replace(' ', ''))
            
            # Average word and sentence lengths
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            
            # Vocabulary richness
            unique_words = len(set(word.lower() for word in words))
            vocabulary_richness = unique_words / len(words) if words else 0
            
            # Paragraph analysis
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            avg_paragraph_length = len(words) / len(paragraphs) if paragraphs else 0
            
            return {
                'word_count': len(words),
                'sentence_count': len(sentences),
                'paragraph_count': len(paragraphs),
                'character_count': char_count,
                'character_count_no_spaces': char_count_no_spaces,
                'avg_word_length': round(avg_word_length, 2),
                'avg_sentence_length': round(avg_sentence_length, 2),
                'avg_paragraph_length': round(avg_paragraph_length, 2),
                'vocabulary_richness': round(vocabulary_richness, 3),
                'unique_words': unique_words
            }
            
        except Exception as e:
            return {'error': f'Text statistics calculation failed: {str(e)}'}
    
    def _analyze_quality_indicators(self, text: str) -> Dict:
        """Analyze quality indicators in the text"""
        try:
            quality_scores = {}
            
            for category, patterns in self.quality_indicators.items():
                total_matches = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    total_matches += matches
                quality_scores[category] = total_matches
            
            # Calculate overall quality score
            total_indicators = sum(quality_scores.values())
            quality_score = min(total_indicators / 10, 1.0)  # Normalize to 0-1
            
            return {
                'professional_language': quality_scores['professional_language'],
                'confidence_indicators': quality_scores['confidence_indicators'],
                'technical_indicators': quality_scores['technical_indicators'],
                'total_indicators': total_indicators,
                'quality_score': round(quality_score, 3)
            }
            
        except Exception as e:
            return {'error': f'Quality indicators analysis failed: {str(e)}'}
    
    def _analyze_structure(self, text: str) -> Dict:
        """Analyze text structure and organization"""
        try:
            # Split into sentences and analyze
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return {'error': 'No sentences found'}
            
            # Sentence length analysis
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_sentence_length = np.mean(sentence_lengths)
            sentence_length_std = np.std(sentence_lengths)
            
            # Sentence length consistency
            length_consistency = 1 - (sentence_length_std / avg_sentence_length) if avg_sentence_length > 0 else 0
            
            # Paragraph structure analysis
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            paragraph_lengths = [len(p.split()) for p in paragraphs]
            avg_paragraph_length = np.mean(paragraph_lengths) if paragraph_lengths else 0
            
            # Structure indicators
            structure_indicators = {
                'has_introduction': self._has_introduction(text),
                'has_conclusion': self._has_conclusion(text),
                'uses_transitions': self._uses_transitions(text),
                'logical_flow': self._assess_logical_flow(sentences)
            }
            
            # Overall structure score
            structure_score = self._calculate_structure_score(
                length_consistency, avg_sentence_length, structure_indicators
            )
            
            return {
                'avg_sentence_length': round(avg_sentence_length, 2),
                'sentence_length_consistency': round(length_consistency, 3),
                'avg_paragraph_length': round(avg_paragraph_length, 2),
                'structure_indicators': structure_indicators,
                'structure_score': round(structure_score, 3)
            }
            
        except Exception as e:
            return {'error': f'Structure analysis failed: {str(e)}'}
    
    def _analyze_coherence(self, text: str) -> Dict:
        """Analyze text coherence and flow"""
        try:
            # Sentence transition analysis
            transition_words = [
                'however', 'therefore', 'furthermore', 'moreover', 'additionally',
                'consequently', 'meanwhile', 'subsequently', 'initially', 'finally',
                'first', 'second', 'third', 'next', 'then', 'also', 'besides'
            ]
            
            transition_count = sum(1 for word in transition_words if word.lower() in text.lower())
            
            # Repetition analysis
            words = text.lower().split()
            word_frequency = {}
            for word in words:
                if len(word) > 3:  # Ignore short words
                    word_frequency[word] = word_frequency.get(word, 0) + 1
            
            # Find overused words
            overused_words = {word: count for word, count in word_frequency.items() if count > 3}
            
            # Coherence score based on transitions and repetition
            coherence_score = min(transition_count / 5, 1.0)  # Normalize transitions
            if overused_words:
                coherence_score *= 0.8  # Penalize repetition
            
            return {
                'transition_count': transition_count,
                'overused_words': overused_words,
                'coherence_score': round(coherence_score, 3),
                'word_frequency': dict(sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:10])
            }
            
        except Exception as e:
            return {'error': f'Coherence analysis failed: {str(e)}'}
    
    def _analyze_professional_language(self, text: str) -> Dict:
        """Analyze professional language usage"""
        try:
            # Professional terminology
            professional_terms = [
                'collaboration', 'implementation', 'optimization', 'methodology',
                'framework', 'architecture', 'infrastructure', 'scalability',
                'efficiency', 'performance', 'reliability', 'integration'
            ]
            
            professional_term_count = sum(1 for term in professional_terms if term.lower() in text.lower())
            
            # Action verbs
            action_verbs = [
                'implemented', 'developed', 'managed', 'coordinated', 'led',
                'achieved', 'delivered', 'optimized', 'streamlined', 'facilitated'
            ]
            
            action_verb_count = sum(1 for verb in action_verbs if verb.lower() in text.lower())
            
            # Technical depth indicators
            technical_indicators = [
                'algorithm', 'data structure', 'database', 'API', 'framework',
                'library', 'tool', 'platform', 'system', 'application'
            ]
            
            technical_count = sum(1 for indicator in technical_indicators if indicator.lower() in text.lower())
            
            # Professional language score
            professional_score = (professional_term_count + action_verb_count + technical_count) / 20
            professional_score = min(professional_score, 1.0)
            
            return {
                'professional_terms': professional_term_count,
                'action_verbs': action_verb_count,
                'technical_indicators': technical_count,
                'professional_score': round(professional_score, 3),
                'professional_level': self._assess_professional_level(professional_score)
            }
            
        except Exception as e:
            return {'error': f'Professional language analysis failed: {str(e)}'}
    
    def _generate_overall_assessment(self, text_stats: Dict, quality_analysis: Dict, 
                                   structure_analysis: Dict, coherence_analysis: Dict, 
                                   professional_analysis: Dict) -> Dict:
        """Generate overall assessment of the transcript"""
        try:
            # Calculate component scores
            word_count = text_stats.get('word_count', 0)
            quality_score = quality_analysis.get('quality_score', 0)
            structure_score = structure_analysis.get('structure_score', 0)
            coherence_score = coherence_analysis.get('coherence_score', 0)
            professional_score = professional_analysis.get('professional_score', 0)
            
            # Weighted overall score
            weights = {
                'quality': 0.3,
                'structure': 0.25,
                'coherence': 0.2,
                'professional': 0.25
            }
            
            overall_score = (
                quality_score * weights['quality'] +
                structure_score * weights['structure'] +
                coherence_score * weights['coherence'] +
                professional_score * weights['professional']
            )
            
            # Determine overall rating
            if overall_score >= 0.8:
                rating = "Excellent"
            elif overall_score >= 0.6:
                rating = "Good"
            elif overall_score >= 0.4:
                rating = "Average"
            else:
                rating = "Needs Improvement"
            
            # Generate specific feedback
            feedback = self._generate_feedback(
                word_count, quality_score, structure_score, coherence_score, professional_score
            )
            
            return {
                'overall_score': round(overall_score, 3),
                'rating': rating,
                'component_scores': {
                    'quality': quality_score,
                    'structure': structure_score,
                    'coherence': coherence_score,
                    'professional': professional_score
                },
                'feedback': feedback,
                'recommendations': self._generate_recommendations(overall_score, feedback)
            }
            
        except Exception as e:
            return {'error': f'Overall assessment generation failed: {str(e)}'}
    
    def _generate_transcript_summary(self, overall_assessment: Dict, quality_analysis: Dict) -> Dict:
        """Generate transcript summary"""
        try:
            if 'error' in overall_assessment:
                return {'error': overall_assessment['error']}
            
            rating = overall_assessment.get('rating', 'Unknown')
            overall_score = overall_assessment.get('overall_score', 0)
            
            # Generate summary description
            if rating == "Excellent":
                description = "Outstanding transcript with excellent structure, professional language, and clear communication"
            elif rating == "Good":
                description = "Good transcript with solid structure and professional language, minor improvements possible"
            elif rating == "Average":
                description = "Average transcript with room for improvement in structure and professional language"
            else:
                description = "Transcript needs significant improvement in multiple areas"
            
            return {
                'rating': rating,
                'score': overall_score,
                'description': description,
                'key_strengths': self._identify_key_strengths(overall_assessment),
                'key_weaknesses': self._identify_key_weaknesses(overall_assessment)
            }
            
        except Exception as e:
            return {'error': f'Summary generation failed: {str(e)}'}
    
    # Helper methods
    def _has_introduction(self, text: str) -> bool:
        """Check if text has a clear introduction"""
        intro_indicators = ['introduction', 'overview', 'summary', 'beginning', 'start']
        first_sentence = text.split('.')[0].lower()
        return any(indicator in first_sentence for indicator in intro_indicators)
    
    def _has_conclusion(self, text: str) -> bool:
        """Check if text has a clear conclusion"""
        conclusion_indicators = ['conclusion', 'summary', 'finally', 'in conclusion', 'to summarize']
        last_sentence = text.split('.')[-1].lower()
        return any(indicator in last_sentence for indicator in conclusion_indicators)
    
    def _uses_transitions(self, text: str) -> bool:
        """Check if text uses transition words"""
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'additionally']
        return any(word in text.lower() for word in transition_words)
    
    def _assess_logical_flow(self, sentences: List[str]) -> float:
        """Assess logical flow of sentences"""
        # Simple heuristic: check for logical connectors and topic consistency
        connectors = ['however', 'therefore', 'furthermore', 'moreover', 'additionally']
        connector_count = sum(1 for sentence in sentences for connector in connectors if connector in sentence.lower())
        return min(connector_count / len(sentences), 1.0) if sentences else 0
    
    def _calculate_structure_score(self, length_consistency: float, avg_sentence_length: float, 
                                 structure_indicators: Dict) -> float:
        """Calculate overall structure score"""
        score = 0
        
        # Length consistency (30%)
        score += length_consistency * 0.3
        
        # Optimal sentence length (20%)
        if 15 <= avg_sentence_length <= 25:
            score += 0.2
        elif 10 <= avg_sentence_length <= 30:
            score += 0.1
        
        # Structure indicators (50%)
        indicator_score = sum(structure_indicators.values()) / len(structure_indicators)
        score += indicator_score * 0.5
        
        return min(score, 1.0)
    
    def _assess_professional_level(self, professional_score: float) -> str:
        """Assess professional language level"""
        if professional_score >= 0.8:
            return "Highly Professional"
        elif professional_score >= 0.6:
            return "Professional"
        elif professional_score >= 0.4:
            return "Semi-Professional"
        else:
            return "Casual"
    
    def _generate_feedback(self, word_count: int, quality_score: float, structure_score: float,
                          coherence_score: float, professional_score: float) -> List[str]:
        """Generate specific feedback"""
        feedback = []
        
        # Word count feedback
        if word_count < 100:
            feedback.append("Response is quite brief - consider providing more detail")
        elif word_count > 500:
            feedback.append("Response is comprehensive - good level of detail")
        
        # Quality feedback
        if quality_score < 0.3:
            feedback.append("Limited use of professional language and technical indicators")
        elif quality_score > 0.7:
            feedback.append("Excellent use of professional language and technical indicators")
        
        # Structure feedback
        if structure_score < 0.4:
            feedback.append("Structure could be improved with better organization")
        elif structure_score > 0.8:
            feedback.append("Well-structured and organized response")
        
        # Coherence feedback
        if coherence_score < 0.4:
            feedback.append("Flow and coherence could be improved with better transitions")
        elif coherence_score > 0.8:
            feedback.append("Excellent flow and logical progression")
        
        return feedback
    
    def _generate_recommendations(self, overall_score: float, feedback: List[str]) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []
        
        if overall_score < 0.4:
            recommendations.extend([
                "Focus on developing professional language skills",
                "Work on structuring responses with clear introduction and conclusion",
                "Practice using transition words to improve flow",
                "Expand technical vocabulary and knowledge"
            ])
        elif overall_score < 0.6:
            recommendations.extend([
                "Continue developing professional communication skills",
                "Practice organizing thoughts before responding",
                "Work on using more technical terminology appropriately"
            ])
        else:
            recommendations.extend([
                "Maintain current level of professional communication",
                "Consider taking on more complex projects to further develop skills"
            ])
        
        return recommendations
    
    def _identify_key_strengths(self, overall_assessment: Dict) -> List[str]:
        """Identify key strengths"""
        strengths = []
        component_scores = overall_assessment.get('component_scores', {})
        
        if component_scores.get('quality', 0) > 0.7:
            strengths.append("Strong professional language usage")
        if component_scores.get('structure', 0) > 0.7:
            strengths.append("Well-structured response")
        if component_scores.get('coherence', 0) > 0.7:
            strengths.append("Clear and logical flow")
        if component_scores.get('professional', 0) > 0.7:
            strengths.append("High level of professionalism")
        
        return strengths
    
    def _identify_key_weaknesses(self, overall_assessment: Dict) -> List[str]:
        """Identify key weaknesses"""
        weaknesses = []
        component_scores = overall_assessment.get('component_scores', {})
        
        if component_scores.get('quality', 0) < 0.4:
            weaknesses.append("Limited professional language usage")
        if component_scores.get('structure', 0) < 0.4:
            weaknesses.append("Poor structure and organization")
        if component_scores.get('coherence', 0) < 0.4:
            weaknesses.append("Unclear flow and transitions")
        if component_scores.get('professional', 0) < 0.4:
            weaknesses.append("Low level of professionalism")
        
        return weaknesses
