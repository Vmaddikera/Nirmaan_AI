import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json

class CandidateAssessor:
    def __init__(self):
        self.assessment_weights = {
            'emotion': 0.25,
            'gaze': 0.25,
            'nlp': 0.30,
            'transcript': 0.20
        }
    
    def assess_candidate(self, emotion_results: Dict, gaze_results: Dict, 
                        nlp_results: Dict, transcript_results: Dict, transcript_text: str = "") -> Dict:
        """
        Comprehensive candidate assessment combining all analysis results
        """
        try:
            # 1. Individual component assessments
            emotion_assessment = self._assess_emotion_component(emotion_results)
            gaze_assessment = self._assess_gaze_component(gaze_results)
            nlp_assessment = self._assess_nlp_component(nlp_results)
            transcript_assessment = self._assess_transcript_component(transcript_results)
            
            # 2. Strengths and weaknesses analysis
            strengths_weaknesses = self._analyze_strengths_weaknesses(
                emotion_assessment, gaze_assessment, nlp_assessment, transcript_assessment
            )
            
            # 3. Extract strengths directly from transcript
            transcript_strengths = self._extract_strengths_from_transcript(transcript_text)
            if transcript_strengths:
                strengths_weaknesses['strengths'].extend(transcript_strengths)
                strengths_weaknesses['strengths'] = list(set(strengths_weaknesses['strengths']))  # Remove duplicates
            
            # Generate insights
            insights = self._generate_insights(emotion_results, gaze_results, nlp_results, transcript_results)
            
            return {
                'component_scores': {
                    'emotion': emotion_assessment['score'],
                    'gaze': gaze_assessment['score'],
                    'nlp': nlp_assessment['score'],
                    'transcript': transcript_assessment['score']
                },
                'strengths': strengths_weaknesses['strengths'],
                'weaknesses': strengths_weaknesses['weaknesses'],
                'insights': insights,
                'detailed_assessment': {
                    'emotion': emotion_assessment,
                    'gaze': gaze_assessment,
                    'nlp': nlp_assessment,
                    'transcript': transcript_assessment
                }
            }
            
        except Exception as e:
            return {'error': f'Candidate assessment failed: {str(e)}'}
    
    def _generate_insights(self, emotion_results, gaze_results, nlp_results, transcript_results):
        """Generate key insights from all analysis results"""
        insights = []
        
        # Emotion insights with more detail
        if 'dominant_emotion' in emotion_results:
            emotion = emotion_results['dominant_emotion']
            emotion_score = emotion_results.get('emotion_score', 0)
            if emotion == 'joy':
                if emotion_score > 0.7:
                    insights.append("Candidate shows strong enthusiasm and positive attitude")
                else:
                    insights.append("Candidate shows enthusiasm and positive attitude")
            elif emotion == 'neutral':
                if emotion_score > 0.6:
                    insights.append("Candidate maintains excellent professional composure")
                else:
                    insights.append("Candidate maintains professional composure")
            elif emotion == 'fear':
                insights.append("Candidate may need confidence building and support")
            elif emotion == 'sadness':
                insights.append("Candidate may be experiencing stress or low confidence")
            elif emotion == 'anger':
                insights.append("Candidate may be frustrated or defensive")
        
        # Gaze insights with more detail
        if 'focus_level' in gaze_results:
            focus = gaze_results['focus_level']
            attention_score = gaze_results.get('attention_score', 0)
            if focus == "High":
                if attention_score > 0.8:
                    insights.append("Excellent attention and focus during interview")
                else:
                    insights.append("Good attention and focus during interview")
            elif focus == "Medium":
                insights.append("Moderate attention and focus - room for improvement")
            elif focus == "Low":
                insights.append("May need to improve attention and engagement")
        
        # NLP insights with skills information
        if 'sentiment' in nlp_results:
            sentiment = nlp_results['sentiment']
            confidence = nlp_results.get('sentiment_confidence', 0)
            if sentiment == 'POSITIVE':
                if confidence > 0.8:
                    insights.append("Strong positive communication and high confidence")
                else:
                    insights.append("Positive communication and confidence")
            elif sentiment == 'NEGATIVE':
                insights.append("Communication tone may need improvement")
            elif sentiment == 'NEUTRAL':
                insights.append("Professional but neutral communication tone")
        
        # Add skills insights if available
        if 'skills' in nlp_results and 'key_skills' in nlp_results['skills']:
            key_skills = nlp_results['skills']['key_skills']
            if key_skills:
                skills_text = ", ".join(key_skills[:3])  # Top 3 skills
                insights.append(f"Key skills identified: {skills_text}")
        
        # Transcript insights with more detail
        if 'response_quality' in transcript_results:
            quality = transcript_results['response_quality']
            communication_score = transcript_results.get('communication_score', 0)
            clarity_score = transcript_results.get('clarity_score', 0)
            
            if quality == "Strong":
                if communication_score > 0.8 and clarity_score > 0.8:
                    insights.append("Excellent communication skills and response quality")
                else:
                    insights.append("Strong communication skills and response quality")
            elif quality == "Average":
                if clarity_score > 0.8:
                    insights.append("Communication could be more structured and detailed")
                else:
                    insights.append("Communication structure and clarity need development")
            elif quality == "Needs Improvement":
                if clarity_score > 0.8:
                    insights.append("Communication structure needs improvement but clarity is good")
                else:
                    insights.append("Communication structure and clarity need development")
        
        # Add overall performance insight
        overall_insight = self._generate_overall_insight(emotion_results, gaze_results, nlp_results, transcript_results)
        if overall_insight:
            insights.append(overall_insight)
        
        return insights
    
    def _generate_overall_insight(self, emotion_results, gaze_results, nlp_results, transcript_results):
        """Generate an overall performance insight"""
        try:
            # Calculate a simple overall performance indicator
            scores = []
            
            if 'emotion_score' in emotion_results:
                scores.append(emotion_results['emotion_score'])
            if 'attention_score' in gaze_results:
                scores.append(gaze_results['attention_score'])
            if 'communication_score' in transcript_results:
                scores.append(transcript_results['communication_score'])
            
            if scores:
                avg_score = sum(scores) / len(scores)
                if avg_score > 0.8:
                    return "Overall excellent performance across multiple assessment areas"
                elif avg_score > 0.6:
                    return "Good overall performance with some areas for improvement"
                elif avg_score > 0.4:
                    return "Moderate performance with several areas needing attention"
                else:
                    return "Performance needs significant improvement across multiple areas"
            
            return None
        except Exception:
            return None
    
    def _assess_emotion_component(self, emotion_results: Dict) -> Dict:
        """Assess emotional intelligence using sophisticated scoring"""
        try:
            if 'error' in emotion_results:
                return {'score': 0, 'assessment': 'Emotion analysis failed', 'details': emotion_results['error']}
            
            details = []
            scores = {}
            
            # 1. EMOTION POSITIVITY SCORE (0-1)
            emotion_scores = emotion_results.get('emotion_scores', {})
            positive_emotions = emotion_scores.get('joy', 0) + emotion_scores.get('surprise', 0) * 0.5
            negative_emotions = emotion_scores.get('anger', 0) + emotion_scores.get('sadness', 0) + emotion_scores.get('fear', 0) + emotion_scores.get('disgust', 0)
            neutral_emotions = emotion_scores.get('neutral', 0)
            
            # Calculate positivity ratio
            total_emotion = positive_emotions + negative_emotions + neutral_emotions
            if total_emotion > 0:
                positivity_score = (positive_emotions + neutral_emotions * 0.5) / total_emotion
            else:
                positivity_score = 0.5  # Default neutral
            
            scores['positivity'] = positivity_score
            details.append(f"Emotional positivity: {positivity_score:.2f}")
            
            # 2. EMOTIONAL BALANCE SCORE (0-1)
            profile = emotion_results.get('emotion_profile', {})
            balance = profile.get('emotional_balance', 0)
            scores['balance'] = balance
            details.append(f"Emotional balance: {balance:.2f}")
            
            # 3. DOMINANT EMOTION APPROPRIATENESS (0-1)
            dominant_emotion = emotion_results.get('dominant_emotion', 'unknown')
            emotion_appropriateness = {
                'joy': 0.9, 'neutral': 0.8, 'surprise': 0.7,
                'sadness': 0.4, 'anger': 0.2, 'fear': 0.3, 'disgust': 0.1
            }
            appropriateness_score = emotion_appropriateness.get(dominant_emotion, 0.5)
            scores['appropriateness'] = appropriateness_score
            details.append(f"Emotion appropriateness: {appropriateness_score:.2f}")
            
            # 4. WEIGHTED OVERALL EMOTION SCORE
            weights = {
                'positivity': 0.40,
                'balance': 0.30,
                'appropriateness': 0.30
            }
            
            overall_score = sum(scores[key] * weights[key] for key in weights.keys())
            normalized_score = min(overall_score, 1.0)
            
            return {
                'score': normalized_score,
                'assessment': self._get_emotion_assessment(normalized_score),
                'details': details,
                'strengths': self._get_emotion_strengths(emotion_results),
                'weaknesses': self._get_emotion_weaknesses(emotion_results)
            }
            
        except Exception as e:
            return {'score': 0, 'assessment': f'Emotion assessment failed: {str(e)}', 'details': []}
    
    def _assess_gaze_component(self, gaze_results: Dict) -> Dict:
        """Assess attention, focus, and engagement through gaze analysis"""
        try:
            if 'error' in gaze_results:
                return {'score': 0, 'assessment': 'Gaze analysis failed', 'details': gaze_results['error']}
            
            score = 0
            details = []
            
            # Analyze focus level
            summary = gaze_results.get('summary', {})
            focus_level = summary.get('focus_level', 'Unknown')
            
            if focus_level == 'High Focus':
                score += 3
                details.append("Excellent focus and attention")
            elif focus_level == 'Moderate Focus':
                score += 2
                details.append("Good focus and attention")
            else:
                score += 1
                details.append("Focus needs improvement")
            
            # Analyze attention quality
            attention_quality = summary.get('attention_quality', 'Unknown')
            if attention_quality == 'Consistent':
                score += 3
                details.append("Consistent attention patterns")
            elif attention_quality == 'Moderately Consistent':
                score += 2
                details.append("Generally consistent attention")
            else:
                score += 1
                details.append("Inconsistent attention patterns")
            
            # Analyze engagement
            engagement = summary.get('engagement', 'Unknown')
            if engagement == 'High Engagement':
                score += 3
                details.append("High engagement level")
            elif engagement == 'Moderate Engagement':
                score += 2
                details.append("Moderate engagement level")
            else:
                score += 1
                details.append("Low engagement level")
            
            # Analyze gaze consistency
            gaze_consistency = summary.get('gaze_consistency', 'Unknown')
            if gaze_consistency == 'Very Consistent':
                score += 1
                details.append("Very consistent gaze patterns")
            elif gaze_consistency == 'Moderately Consistent':
                score += 0.5
                details.append("Moderately consistent gaze patterns")
            
            # Normalize score to 0-1 range (assuming max possible score is around 8-10)
            max_possible_score = 8.0
            normalized_score = min(score / max_possible_score, 1.0)
            
            return {
                'score': normalized_score,
                'assessment': self._get_gaze_assessment(normalized_score),
                'details': details,
                'strengths': self._get_gaze_strengths(gaze_results),
                'weaknesses': self._get_gaze_weaknesses(gaze_results)
            }
            
        except Exception as e:
            return {'score': 0, 'assessment': f'Gaze assessment failed: {str(e)}', 'details': []}
    
    def _assess_nlp_component(self, nlp_results: Dict) -> Dict:
        """Assess communication skills and professional profile"""
        try:
            if 'error' in nlp_results:
                return {'score': 0, 'assessment': 'NLP analysis failed', 'details': nlp_results['error']}
            
            score = 0
            details = []
            
            # Analyze sentiment
            sentiment = nlp_results.get('sentiment', {})
            overall_sentiment = sentiment.get('overall_sentiment', 'Unknown')
            if overall_sentiment == 'POSITIVE':
                score += 2
                details.append("Positive communication tone")
            elif overall_sentiment == 'NEGATIVE':
                score += 0
                details.append("Negative communication tone")
            else:
                score += 1
                details.append("Neutral communication tone")
            
            # Analyze communication quality
            communication = nlp_results.get('communication', {})
            comm_quality = communication.get('communication_quality', 'Unknown')
            if comm_quality == 'Excellent':
                score += 3
                details.append("Excellent communication skills")
            elif comm_quality == 'Good':
                score += 2
                details.append("Good communication skills")
            elif comm_quality == 'Average':
                score += 1
                details.append("Average communication skills")
            else:
                score += 0
                details.append("Communication skills need improvement")
            
            # Analyze skills
            skills = nlp_results.get('skills', {})
            total_skills = skills.get('total_skills', 0)
            if total_skills > 10:
                score += 2
                details.append("Diverse skill set demonstrated")
            elif total_skills > 5:
                score += 1
                details.append("Good technical competency")
            else:
                score += 0
                details.append("Limited skill demonstration")
            
            # Analyze professional profile
            profile = nlp_results.get('profile', {})
            career_level = profile.get('career_level', 'unknown')
            if career_level in ['mid_level', 'senior_level']:
                score += 1
                details.append("Shows professional maturity")
            
            # Analyze vocabulary richness
            vocab_richness = communication.get('vocabulary_richness', 0)
            if vocab_richness > 0.7:
                score += 1
                details.append("Rich vocabulary usage")
            elif vocab_richness > 0.5:
                score += 0.5
                details.append("Good vocabulary usage")
            
            # Normalize score to 0-1 range (assuming max possible score is around 8-10)
            max_possible_score = 8.0
            normalized_score = min(score / max_possible_score, 1.0)
            
            return {
                'score': normalized_score,
                'assessment': self._get_nlp_assessment(normalized_score),
                'details': details,
                'strengths': self._get_nlp_strengths(nlp_results),
                'weaknesses': self._get_nlp_weaknesses(nlp_results)
            }
            
        except Exception as e:
            return {'score': 0, 'assessment': f'NLP assessment failed: {str(e)}', 'details': []}
    
    def _assess_transcript_component(self, transcript_results: Dict) -> Dict:
        """Assess transcript quality and content"""
        try:
            if 'error' in transcript_results:
                return {'score': 0, 'assessment': 'Transcript analysis failed', 'details': transcript_results['error']}
            
            score = 0
            details = []
            
            # Analyze transcript length and structure
            word_count = transcript_results.get('word_count', 0)
            if word_count > 200:
                score += 2
                details.append("Comprehensive response")
            elif word_count > 100:
                score += 1
                details.append("Adequate response length")
            else:
                score += 0
                details.append("Brief response")
            
            # Analyze content quality indicators
            quality_indicators = transcript_results.get('quality_indicators', {})
            professional_language = quality_indicators.get('professional_language', 0)
            if professional_language > 5:
                score += 2
                details.append("Professional language usage")
            elif professional_language > 2:
                score += 1
                details.append("Some professional language")
            
            confidence_indicators = quality_indicators.get('confidence_indicators', 0)
            if confidence_indicators > 3:
                score += 2
                details.append("Confident communication")
            elif confidence_indicators > 1:
                score += 1
                details.append("Moderate confidence")
            
            # Analyze structure and coherence
            structure_score = transcript_results.get('structure_score', 0)
            if structure_score > 0.7:
                score += 2
                details.append("Well-structured response")
            elif structure_score > 0.4:
                score += 1
                details.append("Moderately structured response")
            
            # Normalize score to 0-1 range (assuming max possible score is around 8-10)
            max_possible_score = 8.0
            normalized_score = min(score / max_possible_score, 1.0)
            
            return {
                'score': normalized_score,
                'assessment': self._get_transcript_assessment(normalized_score),
                'details': details,
                'strengths': self._get_transcript_strengths(transcript_results),
                'weaknesses': self._get_transcript_weaknesses(transcript_results)
            }
            
        except Exception as e:
            return {'score': 0, 'assessment': f'Transcript assessment failed: {str(e)}', 'details': []}
    
    def _analyze_strengths_weaknesses(self, emotion_assessment: Dict, gaze_assessment: Dict,
                                    nlp_assessment: Dict, transcript_assessment: Dict) -> Dict:
        """Analyze overall strengths and weaknesses"""
        try:
            strengths = []
            weaknesses = []
            
            
            # Analyze each component and generate strengths/weaknesses based on scores
            components = [
                ('Emotional Intelligence', emotion_assessment),
                ('Attention & Focus', gaze_assessment),
                ('Communication', nlp_assessment),
                ('Content Quality', transcript_assessment)
            ]
            
            for component_name, assessment in components:
                score = assessment.get('score', 0)
                
                if score >= 0.5:
                    # Medium to high performance - add strengths
                    if component_name == 'Emotional Intelligence':
                        strengths.extend([
                            "Strong emotional intelligence and positive attitude",
                            "Good emotional intelligence and self-awareness"
                        ])
                    elif component_name == 'Attention & Focus':
                        strengths.extend([
                            "Excellent attention and focus during interview",
                            "Strong engagement and eye contact"
                        ])
                    elif component_name == 'Communication':
                        strengths.extend([
                            "Clear and effective communication skills",
                            "Strong verbal presentation abilities"
                        ])
                    elif component_name == 'Content Quality':
                        strengths.extend([
                            "Well-structured and professional responses",
                            "High-quality content and technical knowledge"
                        ])
                elif score <= 0.3:
                    # Low performance - add weaknesses
                    if component_name == 'Emotional Intelligence':
                        weaknesses.extend([
                            "Emotional intelligence needs improvement",
                            "May benefit from confidence building"
                        ])
                    elif component_name == 'Attention & Focus':
                        weaknesses.extend([
                            "Attention and focus could be improved",
                            "May need to work on engagement during interviews"
                        ])
                    elif component_name == 'Communication':
                        weaknesses.extend([
                            "Communication skills need development",
                            "Verbal presentation could be more structured"
                        ])
                    elif component_name == 'Content Quality':
                        weaknesses.extend([
                            "Response quality and structure need improvement",
                            "Technical knowledge and content depth could be enhanced"
                        ])
            
            # Add overall assessment based on component scores
            total_high_scores = sum(1 for assessment in [emotion_assessment, gaze_assessment, nlp_assessment, transcript_assessment] if assessment.get('score', 0) >= 0.5)
            total_low_scores = sum(1 for assessment in [emotion_assessment, gaze_assessment, nlp_assessment, transcript_assessment] if assessment.get('score', 0) <= 0.3)
            
            if total_high_scores >= 3:
                strengths.append("Overall excellent performance across multiple areas")
            elif total_high_scores >= 2:
                strengths.append("Good performance in several key areas")
            elif total_low_scores >= 3:
                weaknesses.append("Significant improvement needed across multiple areas")
            elif total_low_scores >= 2:
                weaknesses.append("Moderate performance with several areas needing attention")
            
            return {
                'strengths': list(set(strengths)),
                'weaknesses': list(set(weaknesses))
            }
            
        except Exception as e:
            return {'strengths': [], 'weaknesses': [f'Analysis failed: {str(e)}']}
    
    def _extract_strengths_from_transcript(self, transcript_text: str) -> List[str]:
        """Extract strengths directly from transcript using regex patterns"""
        try:
            import re
            
            if not transcript_text:
                return []
            
            strengths = []
            text_lower = transcript_text.lower()
            
            # Regex patterns for strength indicators
            strength_patterns = [
                # Direct strength statements
                r'my strengths?\s+(?:are|is|include)\s+([^.]*)',
                r'i am strong\s+(?:in|at|with)\s+([^.]*)',
                r'i have\s+(?:strong|excellent|good|great)\s+([^.]*)',
                r'my\s+(?:strength|strengths)\s+(?:is|are)\s+([^.]*)',
                
                # Experience and expertise indicators
                r'extensive experience\s+(?:in|with)\s+([^.]*)',
                r'proven expertise\s+(?:in|with)\s+([^.]*)',
                r'strong background\s+(?:in|with)\s+([^.]*)',
                r'deep knowledge\s+(?:of|in)\s+([^.]*)',
                
                # Skill and ability indicators
                r'i am\s+(?:proficient|skilled|experienced|expert)\s+(?:in|at|with)\s+([^.]*)',
                r'i have\s+(?:developed|gained|acquired)\s+(?:expertise|skills|experience)\s+(?:in|with)\s+([^.]*)',
                r'my\s+(?:expertise|skills|experience)\s+(?:in|with|includes?)\s+([^.]*)',
                
                # Achievement indicators
                r'successfully\s+(?:developed|implemented|delivered|managed)\s+([^.]*)',
                r'effectively\s+(?:led|managed|coordinated|delivered)\s+([^.]*)',
                r'proven track record\s+(?:in|of)\s+([^.]*)',
            ]
            
            for pattern in strength_patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Clean and extract the strength
                    strength = match.strip()
                    if len(strength) > 3 and len(strength) < 100:  # Reasonable length
                        # Clean up the strength text
                        strength = re.sub(r'\s+', ' ', strength)  # Remove extra spaces
                        strength = strength.strip('.,;:')  # Remove trailing punctuation
                        if strength and strength not in strengths:
                            strengths.append(strength.title())
            
            # Also look for specific technical skills mentioned
            tech_skills = [
                'python', 'javascript', 'java', 'react', 'angular', 'vue', 'node.js',
                'django', 'spring boot', 'sql', 'mysql', 'postgresql', 'mongodb',
                'aws', 'azure', 'docker', 'kubernetes', 'machine learning', 'data science'
            ]
            
            for skill in tech_skills:
                if skill in text_lower:
                    skill_strength = f"Technical expertise in {skill.title()}"
                    if skill_strength not in strengths:
                        strengths.append(skill_strength)
            
            return strengths[:10]  # Limit to top 10 strengths
            
        except Exception as e:
            print(f"Error extracting strengths from transcript: {str(e)}")
            return []
    
    
    # Helper methods for individual assessments
    def _get_emotion_assessment(self, score: float) -> str:
        """Get emotion assessment based on 0-1 score"""
        if score >= 0.8:
            return "Excellent emotional intelligence and self-awareness"
        elif score >= 0.6:
            return "Good emotional awareness with room for growth"
        elif score >= 0.4:
            return "Developing emotional intelligence"
        else:
            return "Needs significant emotional development"
    
    def _get_gaze_assessment(self, score: float) -> str:
        """Get gaze assessment based on 0-1 score"""
        if score >= 0.8:
            return "Excellent focus and attention"
        elif score >= 0.6:
            return "Good attention and engagement"
        elif score >= 0.4:
            return "Moderate focus with improvement needed"
        else:
            return "Significant attention and focus issues"
    
    def _get_nlp_assessment(self, score: float) -> str:
        """Get NLP assessment based on 0-1 score"""
        if score >= 0.8:
            return "Excellent communication and professional profile"
        elif score >= 0.6:
            return "Good communication skills"
        elif score >= 0.4:
            return "Average communication with development needed"
        else:
            return "Communication skills need significant improvement"
    
    def _get_transcript_assessment(self, score: float) -> str:
        """Get transcript assessment based on 0-1 score"""
        if score >= 0.8:
            return "Excellent content quality and structure"
        elif score >= 0.6:
            return "Good content with minor improvements needed"
        elif score >= 0.4:
            return "Average content quality"
        else:
            return "Content quality needs significant improvement"
    
    # Helper methods for strengths and weaknesses
    def _get_emotion_strengths(self, emotion_results: Dict) -> List[str]:
        strengths = []
        if emotion_results.get('dominant_emotion', ('', 0))[0] in ['joy', 'neutral']:
            strengths.append("Positive emotional state")
        if emotion_results.get('emotion_score', 0) > 0.7:
            strengths.append("High emotional intelligence")
        return strengths
    
    def _get_emotion_weaknesses(self, emotion_results: Dict) -> List[str]:
        weaknesses = []
        if emotion_results.get('dominant_emotion', ('', 0))[0] in ['anger', 'sadness']:
            weaknesses.append("Concerning emotional patterns")
        if emotion_results.get('emotion_score', 0) < 0.3:
            weaknesses.append("Low emotional intelligence")
        return weaknesses
    
    def _get_gaze_strengths(self, gaze_results: Dict) -> List[str]:
        strengths = []
        summary = gaze_results.get('summary', {})
        if summary.get('focus_level') == 'High Focus':
            strengths.append("Excellent focus and attention")
        if summary.get('engagement') == 'High Engagement':
            strengths.append("High engagement level")
        return strengths
    
    def _get_gaze_weaknesses(self, gaze_results: Dict) -> List[str]:
        weaknesses = []
        summary = gaze_results.get('summary', {})
        if summary.get('focus_level') == 'Low Focus':
            weaknesses.append("Poor focus and attention")
        if summary.get('engagement') == 'Low Engagement':
            weaknesses.append("Low engagement level")
        return weaknesses
    
    def _get_nlp_strengths(self, nlp_results: Dict) -> List[str]:
        strengths = []
        if nlp_results.get('sentiment', {}).get('overall_sentiment') == 'POSITIVE':
            strengths.append("Positive communication tone")
        if nlp_results.get('communication', {}).get('communication_quality') == 'Excellent':
            strengths.append("Excellent communication skills")
        return strengths
    
    def _get_nlp_weaknesses(self, nlp_results: Dict) -> List[str]:
        weaknesses = []
        if nlp_results.get('sentiment', {}).get('overall_sentiment') == 'NEGATIVE':
            weaknesses.append("Negative communication tone")
        if nlp_results.get('communication', {}).get('communication_quality') == 'Needs Improvement':
            weaknesses.append("Poor communication skills")
        return weaknesses
    
    def _get_transcript_strengths(self, transcript_results: Dict) -> List[str]:
        strengths = []
        if transcript_results.get('word_count', 0) > 200:
            strengths.append("Comprehensive response")
        if transcript_results.get('structure_score', 0) > 0.7:
            strengths.append("Well-structured content")
        return strengths
    
    def _get_transcript_weaknesses(self, transcript_results: Dict) -> List[str]:
        weaknesses = []
        if transcript_results.get('word_count', 0) < 100:
            weaknesses.append("Brief response")
        if transcript_results.get('structure_score', 0) < 0.4:
            weaknesses.append("Poor content structure")
        return weaknesses
