import pandas as pd
import numpy as np
import re
import json
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel
)
import torch
from typing import Dict, List, Tuple
import spacy
from datetime import datetime

class NLPAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = None
        self.emotion_analyzer = None
        self.ner_pipeline = None
        self.spacy_model = None
        self.initialized = False
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize all required models with fallback handling"""
        try:
            print("Loading sentiment analysis model...")
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                print("SUCCESS: Sentiment model loaded")
            except Exception as e:
                print(f"WARNING: Sentiment model failed: {str(e)}")
                self.sentiment_analyzer = None
            
            print("Loading emotion analysis model...")
            try:
                self.emotion_analyzer = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
                print("SUCCESS: Emotion model loaded")
            except Exception as e:
                print(f"WARNING: Emotion model failed: {str(e)}")
                self.emotion_analyzer = None
            
            print("Loading NER model...")
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
                print("SUCCESS: NER model loaded")
            except Exception as e:
                print(f"WARNING: NER model failed: {str(e)}")
                self.ner_pipeline = None
            
            print("Loading spaCy model...")
            try:
                self.spacy_model = spacy.load("en_core_web_sm")
                print("SUCCESS: spaCy model loaded")
            except OSError:
                print("WARNING: spaCy model not found, skipping...")
                self.spacy_model = None
            except Exception as e:
                print(f"WARNING: spaCy model failed: {str(e)}")
                self.spacy_model = None
            
            # Mark as initialized if at least one model loaded
            if any([self.sentiment_analyzer, self.emotion_analyzer, self.ner_pipeline]):
                self.initialized = True
                print("SUCCESS: NLP models initialized successfully!")
            else:
                self.initialized = False
                print("WARNING: No models loaded, will use fallback analysis")
            
        except Exception as e:
            print(f"Error initializing models: {str(e)}")
            self.initialized = False
    
    def analyze(self, transcript_text: str) -> Dict:
        """
        Comprehensive NLP analysis using transformers with fallback
        """
        try:
            # Use transcript text for analysis
            combined_text = transcript_text.strip()
            
            # Initialize results with defaults
            sentiment_results = {'overall_sentiment': 'Neutral', 'confidence': 0.5}
            emotion_results = {'dominant_emotion': 'neutral', 'confidence': 0.5}
            ner_results = {'entities': {}, 'total_entities': 0}
            skills_results = {'key_skills': [], 'total_skills': 0}
            communication_analysis = {'communication_quality': 'Good'}
            profile_analysis = {'career_level': 'mid_level', 'industry': 'technology'}
            
            # 1. Sentiment Analysis (with fallback)
            if self.sentiment_analyzer:
                sentiment_results = self._analyze_sentiment(combined_text)
            else:
                sentiment_results = self._fallback_sentiment_analysis(combined_text)
            
            # 2. Emotion Analysis (with fallback)
            if self.emotion_analyzer:
                emotion_results = self._analyze_emotions(combined_text)
            else:
                emotion_results = self._fallback_emotion_analysis(combined_text)
            
            # 3. Named Entity Recognition (with fallback)
            if self.ner_pipeline:
                ner_results = self._extract_entities(combined_text)
            else:
                ner_results = {'entities': {}, 'total_entities': 0}
            
            # 4. Key Skills and Competencies Extraction (always works)
            # Use both regex and LLM-based methods for comprehensive extraction
            regex_skills = self._extract_skills_and_competencies(combined_text)
            llm_skills = self._extract_skills_semantic(combined_text)
            skills_results = self._combine_skills_extraction(regex_skills, llm_skills)
            
            # 5. Communication Quality Analysis (always works)
            communication_analysis = self._analyze_communication_quality(combined_text)
            
            # 6. Professional Profile Analysis (always works)
            profile_analysis = self._analyze_professional_profile(combined_text)
            
            # Extract key information for detailed display
            sentiment_label = sentiment_results.get('overall_sentiment', 'Neutral')
            sentiment_confidence = sentiment_results.get('confidence', 0.5)
            key_skills = skills_results.get('key_skills', [])[:3]  # Top 3 skills
            
            return {
                'sentiment': sentiment_label,
                'sentiment_confidence': sentiment_confidence,
                'key_skills': key_skills,
                'sentiment_details': sentiment_results,
                'emotions': emotion_results,
                'entities': ner_results,
                'skills': skills_results,
                'communication': communication_analysis,
                'profile': profile_analysis,
                'summary': self._generate_nlp_summary(sentiment_results, emotion_results, skills_results, communication_analysis)
            }
            
        except Exception as e:
            return {'error': f'NLP analysis failed: {str(e)}'}
    
    def _fallback_sentiment_analysis(self, text: str) -> Dict:
        """Fallback sentiment analysis using keyword matching"""
        try:
            positive_words = ['strong', 'experience', 'passion', 'innovative', 'excellent', 'good', 'great', 'confident', 'successful', 'achieved']
            negative_words = ['weak', 'poor', 'bad', 'terrible', 'unconfident', 'nervous', 'struggle', 'difficult', 'challenging']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                return {'overall_sentiment': 'POSITIVE', 'confidence': 0.7}
            elif negative_count > positive_count:
                return {'overall_sentiment': 'NEGATIVE', 'confidence': 0.7}
            else:
                return {'overall_sentiment': 'NEUTRAL', 'confidence': 0.5}
        except Exception as e:
            return {'overall_sentiment': 'NEUTRAL', 'confidence': 0.5}
    
    def _fallback_emotion_analysis(self, text: str) -> Dict:
        """Fallback emotion analysis using keyword matching"""
        try:
            emotion_keywords = {
                'joy': ['excited', 'happy', 'enthusiastic', 'passionate', 'love'],
                'fear': ['nervous', 'anxious', 'worried', 'concerned', 'scared'],
                'sadness': ['disappointed', 'sad', 'unfortunate', 'regret'],
                'anger': ['frustrated', 'angry', 'annoyed', 'upset']
            }
            
            text_lower = text.lower()
            emotion_scores = {}
            for emotion, keywords in emotion_keywords.items():
                emotion_scores[emotion] = sum(1 for keyword in keywords if keyword in text_lower)
            
            dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            if dominant_emotion[1] > 0:
                return {'dominant_emotion': dominant_emotion[0], 'confidence': 0.6}
            else:
                return {'dominant_emotion': 'neutral', 'confidence': 0.5}
        except Exception as e:
            return {'dominant_emotion': 'neutral', 'confidence': 0.5}
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment using RoBERTa"""
        try:
            results = self.sentiment_analyzer(text)
            
            # Extract the most confident sentiment
            best_sentiment = max(results[0], key=lambda x: x['score'])
            
            return {
                'overall_sentiment': best_sentiment['label'],
                'confidence': best_sentiment['score'],
                'all_scores': results[0]
            }
        except Exception as e:
            return {'error': f'Sentiment analysis failed: {str(e)}'}
    
    def _analyze_emotions(self, text: str) -> Dict:
        """Analyze emotions using emotion classification model"""
        try:
            results = self.emotion_analyzer(text)
            
            # Extract the most confident emotion
            best_emotion = max(results[0], key=lambda x: x['score'])
            
            return {
                'dominant_emotion': best_emotion['label'],
                'confidence': best_emotion['score'],
                'all_emotions': results[0]
            }
        except Exception as e:
            return {'error': f'Emotion analysis failed: {str(e)}'}
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract named entities using BERT NER"""
        try:
            entities = self.ner_pipeline(text)
            
            # Group entities by type
            entity_types = {}
            for entity in entities:
                entity_type = entity['entity_group']
                if entity_type not in entity_types:
                    entity_types[entity_type] = []
                entity_types[entity_type].append({
                    'text': entity['word'],
                    'confidence': entity['score']
                })
            
            return {
                'entities': entity_types,
                'total_entities': len(entities),
                'entity_types': list(entity_types.keys())
            }
        except Exception as e:
            return {'error': f'Entity extraction failed: {str(e)}'}
    
    def _extract_skills_and_competencies(self, text: str) -> Dict:
        """Extract skills using simple and reliable regex patterns"""
        try:
            extracted_skills = {
                'technical_skills': [],
                'soft_skills': [],
                'education': [],
                'experience': []
            }
            
            text_lower = text.lower()
            
            # Comprehensive technical skills patterns
            tech_patterns = [
                # Development types
                'software development', 'web development', 'mobile development', 'full stack development',
                'backend development', 'frontend development', 'full-stack development',
                'programming', 'coding', 'software engineering', 'application development',
                
                # Programming languages
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'php', 'ruby',
                'swift', 'kotlin', 'scala', 'r', 'matlab',
                
                # Frameworks and libraries
                'react', 'angular', 'vue', 'node.js', 'nodejs', 'django', 'flask', 'spring', 'express',
                'laravel', 'rails', 'asp.net', 'jquery', 'bootstrap', 'tailwind',
                
                # Databases
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'sqlite', 'oracle',
                'sql server', 'cassandra', 'dynamodb',
                
                # Cloud and DevOps
                'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins', 'git', 'github',
                'gitlab', 'ci/cd', 'terraform', 'ansible', 'devops', 'cloud computing',
                
                # AI/ML
                'machine learning', 'data science', 'artificial intelligence', 'deep learning',
                'neural networks', 'nlp', 'natural language processing', 'computer vision',
                'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
                
                # Other technical terms
                'algorithms', 'data structures', 'system architecture', 'microservices',
                'api development', 'rest api', 'graphql', 'testing', 'debugging', 'agile', 'scrum'
            ]
            
            # Soft skills patterns
            soft_patterns = [
                'leadership', 'team leadership', 'people management', 'team management',
                'communication', 'verbal communication', 'written communication', 'public speaking',
                'teamwork', 'collaboration', 'cross-functional collaboration', 'stakeholder management',
                'problem solving', 'critical thinking', 'analytical thinking', 'creative thinking',
                'creativity', 'innovation', 'adaptability', 'flexibility', 'time management',
                'project management', 'mentoring', 'supervision', 'coaching', 'presentation',
                'negotiation', 'conflict resolution', 'emotional intelligence', 'interpersonal skills'
            ]
            
            # Education patterns
            edu_patterns = [
                'bachelor', 'master', 'phd', 'doctorate', 'associate', 'diploma', 'certification',
                'university', 'college', 'institute', 'school', 'computer science', 'software engineering',
                'information technology', 'data science', 'engineering', 'mathematics', 'statistics',
                'business', 'management', 'mba', 'degree', 'graduate', 'undergraduate'
            ]
            
            # Experience patterns
            exp_patterns = [
                'extensive experience', 'years of experience', 'proven track record', 'successful projects',
                'multiple projects', 'various projects', 'diverse projects', 'team collaboration',
                'client interaction', 'stakeholder engagement', 'cross-functional teams',
                'leadership experience', 'management experience', 'technical expertise',
                'professional experience', 'industry experience', 'relevant experience'
            ]
            
            # Extract skills using simple pattern matching
            for pattern in tech_patterns:
                if pattern in text_lower:
                    extracted_skills['technical_skills'].append(pattern.title())
            
            for pattern in soft_patterns:
                if pattern in text_lower:
                    extracted_skills['soft_skills'].append(pattern.title())
            
            for pattern in edu_patterns:
                if pattern in text_lower:
                    extracted_skills['education'].append(pattern.title())
            
            for pattern in exp_patterns:
                if pattern in text_lower:
                    extracted_skills['experience'].append(pattern.title())
            
            # Context-based extraction for compound phrases
            import re
            
            # Look for "experience in X" patterns
            experience_pattern = r'experience in ([^,.\n]+)'
            matches = re.findall(experience_pattern, text_lower)
            for match in matches:
                skill = match.strip()
                if len(skill) > 3:
                    if any(tech_word in skill for tech_word in ['development', 'programming', 'engineering', 'software', 'web', 'mobile', 'data', 'cloud']):
                        extracted_skills['technical_skills'].append(skill.title())
                    elif any(soft_word in skill for soft_word in ['leadership', 'communication', 'management', 'teamwork', 'collaboration']):
                        extracted_skills['soft_skills'].append(skill.title())
            
            # Look for "skills in X" patterns
            skills_pattern = r'skills in ([^,.\n]+)'
            matches = re.findall(skills_pattern, text_lower)
            for match in matches:
                skill = match.strip()
                if len(skill) > 3:
                    extracted_skills['technical_skills'].append(skill.title())
            
            # Look for "background in X" patterns
            background_pattern = r'background in ([^,.\n]+)'
            matches = re.findall(background_pattern, text_lower)
            for match in matches:
                skill = match.strip()
                if len(skill) > 3:
                    if any(tech_word in skill for tech_word in ['development', 'programming', 'engineering', 'software']):
                        extracted_skills['technical_skills'].append(skill.title())
            
            # Deduplicate and clean
            for category in extracted_skills:
                extracted_skills[category] = list(set([skill.strip() for skill in extracted_skills[category] if skill.strip()]))
            
            return extracted_skills
            
        except Exception as e:
            print(f"Skills extraction failed: {str(e)}")
            return {'technical_skills': [], 'soft_skills': [], 'education': [], 'experience': []}
    
    def _combine_skills_extraction(self, regex_skills: Dict, llm_skills: Dict) -> Dict:
        """Combine regex and LLM-based skills extraction for comprehensive results"""
        try:
            combined_skills = {
                'technical_skills': [],
                'soft_skills': [],
                'education': [],
                'experience': []
            }
            
            # Combine skills from both methods
            for category in combined_skills.keys():
                regex_list = regex_skills.get(category, [])
                llm_list = llm_skills.get(category, [])
                
                # Merge and deduplicate
                all_skills = list(set(regex_list + llm_list))
                
                # Clean and filter skills
                cleaned_skills = []
                for skill in all_skills:
                    if skill and len(skill.strip()) > 2:
                        cleaned_skill = skill.strip().title()
                        if cleaned_skill not in cleaned_skills:
                            cleaned_skills.append(cleaned_skill)
                
                combined_skills[category] = cleaned_skills
            
            # Calculate total skills and add metadata
            total_skills = sum(len(skills) for skills in combined_skills.values())
            combined_skills['total_skills'] = total_skills
            combined_skills['extraction_methods'] = ['regex', 'llm']
            combined_skills['key_skills'] = self._get_key_skills(combined_skills)
            
            return combined_skills
            
        except Exception as e:
            print(f"Skills combination failed: {str(e)}")
            # Fallback to regex results
            regex_skills['total_skills'] = sum(len(skills) for skills in regex_skills.values())
            regex_skills['extraction_methods'] = ['regex']
            regex_skills['key_skills'] = self._get_key_skills(regex_skills)
            return regex_skills
    
    def _get_key_skills(self, skills_dict: Dict) -> List[str]:
        """Extract top key skills for display"""
        try:
            key_skills = []
            
            # Prioritize technical skills
            tech_skills = skills_dict.get('technical_skills', [])
            if tech_skills:
                key_skills.extend(tech_skills[:3])  # Top 3 technical skills
            
            # Add soft skills if we have space
            soft_skills = skills_dict.get('soft_skills', [])
            if soft_skills and len(key_skills) < 5:
                key_skills.extend(soft_skills[:2])  # Top 2 soft skills
            
            # Add education if we have space
            edu_skills = skills_dict.get('education', [])
            if edu_skills and len(key_skills) < 5:
                key_skills.extend(edu_skills[:1])  # Top 1 education
            
            return key_skills[:5]  # Max 5 key skills
            
        except Exception as e:
            return []
    
    def _extract_skills_semantic(self, text: str) -> Dict:
        """Extract skills using pre-trained LLM models (production method)"""
        try:
            skills = {'technical_skills': [], 'soft_skills': [], 'education': [], 'experience': []}
            
            # Method 1: Use pre-trained NER models (BERT-based)
            if self.ner_pipeline:
                try:
                    entities = self.ner_pipeline(text)
                    
                    for entity in entities:
                        entity_text = entity['word'].lower()
                        entity_label = entity['entity_group']
                        confidence = entity.get('score', 0.5)
                        
                        # Use medium-confidence entities for better coverage
                        if confidence > 0.5:
                            if entity_label in ['MISC', 'ORG', 'TECHNOLOGY', 'PERSON']:
                                # Technical skills detection
                                if any(tech_word in entity_text for tech_word in 
                                    ['python', 'java', 'javascript', 'react', 'angular', 'vue', 'node', 'sql', 'mongodb', 'aws', 'azure', 'docker', 'kubernetes', 'development', 'programming', 'engineering', 'software', 'web', 'mobile', 'cloud', 'data', 'machine', 'learning', 'ai', 'artificial', 'intelligence']):
                                    skills['technical_skills'].append(entity['word'])
                                # Soft skills detection
                                elif any(soft_word in entity_text for soft_word in 
                                    ['leadership', 'communication', 'teamwork', 'collaboration', 'management', 'problem', 'solving', 'critical', 'thinking', 'creativity', 'innovation', 'adaptability', 'flexibility', 'mentoring', 'coaching', 'presentation', 'negotiation']):
                                    skills['soft_skills'].append(entity['word'])
                                # Education detection
                                elif any(edu_word in entity_text for edu_word in 
                                    ['university', 'college', 'degree', 'bachelor', 'master', 'phd', 'doctorate', 'computer', 'science', 'engineering', 'mathematics', 'statistics', 'business', 'mba']):
                                    skills['education'].append(entity['word'])
                except Exception as e:
                    print(f"NER extraction failed: {str(e)}")
            
            # Method 2: Use spaCy's pre-trained models
            if self.spacy_model:
                try:
                    doc = self.spacy_model(text)
                    
                    for ent in doc.ents:
                        if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'TECHNOLOGY', 'WORK_OF_ART', 'GPE']:
                            ent_text = ent.text.lower()
                            # Technical skills detection
                            if any(tech_term in ent_text for tech_term in 
                                ['development', 'programming', 'engineering', 'software', 'web', 'mobile', 'cloud', 'data', 'machine', 'learning', 'artificial', 'intelligence', 'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node', 'sql', 'mongodb', 'aws', 'azure', 'docker', 'kubernetes']):
                                skills['technical_skills'].append(ent.text)
                            # Soft skills detection
                            elif any(soft_term in ent_text for soft_term in 
                                ['leadership', 'communication', 'team', 'collaboration', 'management', 'presentation', 'mentoring', 'coaching', 'problem', 'solving', 'critical', 'thinking', 'creativity', 'innovation']):
                                skills['soft_skills'].append(ent.text)
                            # Education detection
                            elif any(edu_term in ent_text for edu_term in 
                                ['university', 'college', 'degree', 'bachelor', 'master', 'phd', 'computer', 'science', 'engineering', 'mathematics', 'statistics', 'business', 'mba']):
                                skills['education'].append(ent.text)
                except Exception as e:
                    print(f"spaCy extraction failed: {str(e)}")
            
            # Method 3: Advanced context-aware extraction
            try:
                advanced_skills = self._extract_skills_advanced_llm(text)
                for category, skill_list in advanced_skills.items():
                    skills[category].extend(skill_list)
            except Exception as e:
                print(f"Advanced LLM extraction failed: {str(e)}")
            
            # Clean and deduplicate results
            for category in skills:
                skills[category] = list(set([skill.strip() for skill in skills[category] if skill.strip()]))
            
            return skills
            
        except Exception as e:
            print(f"LLM-based extraction failed: {str(e)}")
            return {'technical_skills': [], 'soft_skills': [], 'education': [], 'experience': []}
    
    def _extract_skills_advanced_llm(self, text: str) -> Dict:
        """Advanced LLM-based skill extraction (production-ready)"""
        try:
            # In production, this would use:
            # 1. RoBERTa for NER: microsoft/DialoGPT-medium
            # 2. BERT for classification: bert-base-uncased
            # 3. Sentence transformers for similarity: all-MiniLM-L6-v2
            
            # For now, use the existing models more intelligently
            skills = {'technical_skills': [], 'soft_skills': [], 'education': [], 'experience': []}
            
            # Robust context-aware extraction - handles variety of candidate expressions
            sentences = text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # Comprehensive skill indicators - covers all ways candidates express skills
                skill_indicators = [
                    # Experience indicators
                    'experience in', 'expertise in', 'knowledge of', 'skills in', 'background in', 'proficiency in',
                    'experienced in', 'skilled in', 'proficient in', 'competent in', 'capable in', 'adept in',
                    'familiar with', 'comfortable with', 'strong in', 'excellent in', 'good at', 'specialized in',
                    'focused on', 'concentrated in', 'dedicated to', 'passionate about', 'interested in',
                    
                    # Achievement indicators
                    'successfully', 'effectively', 'efficiently', 'consistently', 'reliably', 'proven track record',
                    'demonstrated', 'achieved', 'accomplished', 'delivered', 'implemented', 'developed', 'created',
                    'built', 'designed', 'architected', 'optimized', 'streamlined', 'improved', 'enhanced',
                    
                    # Leadership indicators
                    'led', 'managed', 'supervised', 'mentored', 'guided', 'directed', 'coordinated', 'facilitated',
                    'headed', 'oversaw', 'administered', 'governed', 'controlled', 'handled', 'took charge of',
                    
                    # Collaboration indicators
                    'collaborated', 'worked with', 'partnered with', 'teamed up with', 'cooperated with',
                    'interacted with', 'communicated with', 'liaised with', 'coordinated with',
                    
                    # Learning indicators
                    'learned', 'studied', 'trained in', 'certified in', 'qualified in', 'educated in',
                    'graduated with', 'completed', 'finished', 'mastered', 'acquired', 'gained',
                    
                    # Project indicators
                    'worked on', 'contributed to', 'participated in', 'involved in', 'engaged in',
                    'part of', 'member of', 'team member', 'project member', 'key contributor',
                    
                    # Time-based indicators
                    'years of', 'months of', 'extensive', 'long-term', 'short-term', 'recent',
                    'previous', 'past', 'current', 'ongoing', 'continuous', 'sustained',
                    
                    # Intensity indicators
                    'deep', 'thorough', 'comprehensive', 'extensive', 'broad', 'wide-ranging',
                    'specialized', 'focused', 'concentrated', 'detailed', 'in-depth', 'advanced',
                    
                    # Confidence indicators
                    'confident in', 'comfortable with', 'strong background in', 'solid foundation in',
                    'well-versed in', 'knowledgeable about', 'informed about', 'aware of',
                    
                    # Action indicators
                    'utilized', 'applied', 'used', 'employed', 'leveraged', 'harnessed',
                    'exploited', 'took advantage of', 'benefited from', 'capitalized on'
                ]
                
                # Look for any skill indicator in the sentence
                found_indicators = []
                for indicator in skill_indicators:
                    if indicator in sentence.lower():
                        found_indicators.append(indicator)
                
                if found_indicators:
                    # Extract skills using multiple methods
                    extracted_skills_from_sentence = self._extract_skills_from_context(sentence, found_indicators)
                    
                    # Add to appropriate categories
                    for category, skill_list in extracted_skills_from_sentence.items():
                        skills[category].extend(skill_list)
            
            return skills
            
        except Exception as e:
            print(f"Advanced LLM extraction failed: {str(e)}")
            return {'technical_skills': [], 'soft_skills': [], 'education': [], 'experience': []}
    
    def _extract_skills_from_context(self, sentence: str, found_indicators: list) -> Dict:
        """Extract skills from sentence context using multiple robust methods"""
        try:
            skills = {'technical_skills': [], 'soft_skills': [], 'education': [], 'experience': []}
            sentence_lower = sentence.lower()
            
            # Method 1: Direct extraction after indicators
            for indicator in found_indicators:
                if indicator in sentence_lower:
                    # Extract text after the indicator
                    parts = sentence_lower.split(indicator)
                    if len(parts) > 1:
                        skill_text = parts[1].strip()
                        # Clean up the skill text
                        skill_text = self._clean_skill_text(skill_text)
                        if skill_text:
                            categorized_skill = self._categorize_skill(skill_text)
                            if categorized_skill:
                                skills[categorized_skill['category']].append(categorized_skill['skill'])
            
            # Method 2: Extract skills from action verbs
            action_verbs = ['developed', 'created', 'built', 'designed', 'implemented', 'managed', 'led', 'coordinated']
            for verb in action_verbs:
                if verb in sentence_lower:
                    # Extract the object of the action
                    verb_index = sentence_lower.find(verb)
                    after_verb = sentence[verb_index + len(verb):].strip()
                    # Clean and categorize
                    skill_text = self._clean_skill_text(after_verb)
                    if skill_text:
                        categorized_skill = self._categorize_skill(skill_text)
                        if categorized_skill:
                            skills[categorized_skill['category']].append(categorized_skill['skill'])
            
            # Method 3: Extract from prepositional phrases
            prepositions = ['in', 'with', 'using', 'through', 'via', 'by means of']
            for prep in prepositions:
                if f' {prep} ' in sentence_lower:
                    # Extract what comes after the preposition
                    prep_parts = sentence_lower.split(f' {prep} ')
                    if len(prep_parts) > 1:
                        skill_text = prep_parts[1].strip()
                        skill_text = self._clean_skill_text(skill_text)
                        if skill_text:
                            categorized_skill = self._categorize_skill(skill_text)
                            if categorized_skill:
                                skills[categorized_skill['category']].append(categorized_skill['skill'])
            
            # Method 4: Extract from compound sentences
            connectors = ['and', 'as well as', 'along with', 'together with', 'plus', 'including']
            for connector in connectors:
                if f' {connector} ' in sentence_lower:
                    # Extract skills from both sides of the connector
                    parts = sentence_lower.split(f' {connector} ')
                    for part in parts:
                        skill_text = self._clean_skill_text(part)
                        if skill_text:
                            categorized_skill = self._categorize_skill(skill_text)
                            if categorized_skill:
                                skills[categorized_skill['category']].append(categorized_skill['skill'])
            
            # Method 5: Extract from time-based expressions
            time_expressions = ['years of', 'months of', 'extensive', 'long-term', 'short-term']
            for time_expr in time_expressions:
                if time_expr in sentence_lower:
                    # Extract what comes after the time expression
                    time_index = sentence_lower.find(time_expr)
                    after_time = sentence[time_index + len(time_expr):].strip()
                    skill_text = self._clean_skill_text(after_time)
                    if skill_text:
                        categorized_skill = self._categorize_skill(skill_text)
                        if categorized_skill:
                            skills[categorized_skill['category']].append(categorized_skill['skill'])
            
            return skills
            
        except Exception as e:
            print(f"Context skill extraction failed: {str(e)}")
            return {'technical_skills': [], 'soft_skills': [], 'education': [], 'experience': []}
    
    def _clean_skill_text(self, skill_text: str) -> str:
        """Clean and normalize skill text"""
        try:
            if not skill_text:
                return ""
            
            # Remove common endings and punctuation
            skill_text = skill_text.split(',')[0].split('.')[0].split(';')[0].split('and')[0].strip()
            
            # Remove common filler words
            filler_words = ['the', 'a', 'an', 'and', 'or', 'but', 'with', 'for', 'in', 'on', 'at', 'to', 'from']
            words = skill_text.split()
            cleaned_words = [word for word in words if word.lower() not in filler_words]
            
            # Rejoin and limit length
            cleaned_text = ' '.join(cleaned_words[:5])  # Max 5 words
            
            # Only return if it's meaningful (more than 2 characters)
            if len(cleaned_text) > 2:
                return cleaned_text
            
            return ""
            
        except Exception as e:
            return ""
    
    def _categorize_skill(self, skill_text: str) -> Dict:
        """Categorize skill into appropriate category"""
        try:
            skill_lower = skill_text.lower()
            
            # Technical skills patterns
            tech_patterns = [
                'development', 'programming', 'engineering', 'software', 'web', 'mobile', 'cloud', 'data',
                'machine learning', 'artificial intelligence', 'python', 'java', 'javascript', 'react', 'angular',
                'vue', 'node', 'sql', 'mongodb', 'aws', 'azure', 'docker', 'kubernetes', 'git', 'github',
                'algorithms', 'database', 'backend', 'frontend', 'full stack', 'devops', 'testing', 'debugging'
            ]
            
            # Soft skills patterns
            soft_patterns = [
                'leadership', 'communication', 'teamwork', 'collaboration', 'management', 'presentation',
                'problem solving', 'critical thinking', 'creativity', 'adaptability', 'time management',
                'mentoring', 'supervision', 'coordination', 'facilitation', 'negotiation', 'stakeholder'
            ]
            
            # Education patterns
            edu_patterns = [
                'university', 'college', 'degree', 'bachelor', 'master', 'phd', 'doctorate', 'diploma',
                'certification', 'computer science', 'engineering', 'mathematics', 'statistics', 'business',
                'information technology', 'data science', 'software engineering'
            ]
            
            # Experience patterns
            exp_patterns = [
                'experience', 'background', 'track record', 'projects', 'initiatives', 'programs',
                'years of', 'months of', 'extensive', 'proven', 'successful', 'multiple', 'various'
            ]
            
            # Check for technical skills
            if any(pattern in skill_lower for pattern in tech_patterns):
                return {'category': 'technical_skills', 'skill': skill_text}
            
            # Check for soft skills
            if any(pattern in skill_lower for pattern in soft_patterns):
                return {'category': 'soft_skills', 'skill': skill_text}
            
            # Check for education
            if any(pattern in skill_lower for pattern in edu_patterns):
                return {'category': 'education', 'skill': skill_text}
            
            # Check for experience
            if any(pattern in skill_lower for pattern in exp_patterns):
                return {'category': 'experience', 'skill': skill_text}
            
            # Default to technical skills if it contains any technical terms
            if any(term in skill_lower for term in ['skill', 'ability', 'capability', 'competency']):
                return {'category': 'technical_skills', 'skill': skill_text}
            
            return None
            
        except Exception as e:
            return None
    
    def _extract_skills_fallback(self, text: str) -> Dict:
        """Fallback regex-based extraction (not production-ready)"""
        try:
            # Improved patterns as fallback
            skill_patterns = {
                'technical_skills': [
                    r'\b(?:software development|web development|mobile development|programming|coding|python|java|javascript)\b',
                    r'\b(?:machine learning|data science|artificial intelligence|cloud computing|aws|azure)\b'
                ],
                'soft_skills': [
                    r'\b(?:leadership|teamwork|communication|problem solving|critical thinking|collaboration)\b'
                ],
                'education': [
                    r'\b(?:bachelor|master|phd|degree|university|college|computer science|engineering)\b'
                ],
                'experience': [
                    r'\b(?:extensive experience|years of experience|proven track record|successful projects)\b'
                ]
            }
            
            extracted_skills = {}
            for category, patterns in skill_patterns.items():
                matches = []
                for pattern in patterns:
                    found = re.findall(pattern, text, re.IGNORECASE)
                    matches.extend(found)
                extracted_skills[category] = list(set(matches))
            
            return extracted_skills
            
        except Exception as e:
            return {'error': f'Skills extraction failed: {str(e)}'}
    
    def _analyze_communication_quality(self, text: str) -> Dict:
        """Analyze communication quality and style"""
        try:
            # Basic text statistics
            word_count = len(text.split())
            sentence_count = len(re.split(r'[.!?]+', text))
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Vocabulary richness
            unique_words = len(set(text.lower().split()))
            vocabulary_richness = unique_words / word_count if word_count > 0 else 0
            
            # Professional language indicators
            professional_indicators = [
                r'\b(?:implemented|developed|managed|coordinated|led|achieved|delivered|optimized|streamlined)\b',
                r'\b(?:collaborated|facilitated|mentored|supervised|trained|guided)\b',
                r'\b(?:analyzed|evaluated|assessed|reviewed|monitored|tracked)\b'
            ]
            
            professional_score = 0
            for pattern in professional_indicators:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                professional_score += matches
            
            # Confidence indicators
            confidence_indicators = [
                r'\b(?:successfully|effectively|efficiently|significantly|substantially)\b',
                r'\b(?:expertise|proficiency|mastery|advanced|specialized)\b'
            ]
            
            confidence_score = 0
            for pattern in confidence_indicators:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                confidence_score += matches
            
            return {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length,
                'vocabulary_richness': vocabulary_richness,
                'professional_score': professional_score,
                'confidence_score': confidence_score,
                'communication_quality': self._assess_communication_quality(
                    avg_sentence_length, vocabulary_richness, professional_score, confidence_score
                )
            }
            
        except Exception as e:
            return {'error': f'Communication analysis failed: {str(e)}'}
    
    def _assess_communication_quality(self, avg_sentence_length, vocabulary_richness, professional_score, confidence_score):
        """Assess overall communication quality"""
        quality_score = 0
        
        # Sentence length (optimal range: 15-25 words)
        if 15 <= avg_sentence_length <= 25:
            quality_score += 2
        elif 10 <= avg_sentence_length <= 30:
            quality_score += 1
        
        # Vocabulary richness (higher is better)
        if vocabulary_richness > 0.7:
            quality_score += 2
        elif vocabulary_richness > 0.5:
            quality_score += 1
        
        # Professional language
        if professional_score > 5:
            quality_score += 2
        elif professional_score > 2:
            quality_score += 1
        
        # Confidence indicators
        if confidence_score > 3:
            quality_score += 2
        elif confidence_score > 1:
            quality_score += 1
        
        if quality_score >= 6:
            return "Excellent"
        elif quality_score >= 4:
            return "Good"
        elif quality_score >= 2:
            return "Average"
        else:
            return "Needs Improvement"
    
    def _analyze_professional_profile(self, text: str) -> Dict:
        """Analyze professional profile and career indicators"""
        try:
            # Career level indicators
            career_indicators = {
                'entry_level': [
                    r'\b(?:fresh graduate|recent graduate|entry level|junior|intern|trainee)\b',
                    r'\b(?:first job|starting career|beginning|new to)\b'
                ],
                'mid_level': [
                    r'\b(?:3-5 years|mid level|experienced|senior|lead|team lead)\b',
                    r'\b(?:managed|supervised|coordinated|mentored)\b'
                ],
                'senior_level': [
                    r'\b(?:5\+ years|senior|principal|architect|director|manager|head of)\b',
                    r'\b(?:strategic|executive|leadership|vision|strategy)\b'
                ]
            }
            
            career_level_scores = {}
            for level, patterns in career_indicators.items():
                score = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    score += matches
                career_level_scores[level] = score
            
            # Determine likely career level
            likely_level = max(career_level_scores.items(), key=lambda x: x[1])
            
            # Industry indicators
            industry_indicators = {
                'technology': [
                    r'\b(?:software|tech|IT|computer|programming|coding|development|engineering)\b',
                    r'\b(?:startup|fintech|edtech|healthtech|AI|ML|data science)\b'
                ],
                'finance': [
                    r'\b(?:finance|banking|investment|trading|financial|accounting|audit)\b',
                    r'\b(?:risk management|compliance|treasury|capital markets)\b'
                ],
                'consulting': [
                    r'\b(?:consulting|advisory|strategy|management consulting|business)\b',
                    r'\b(?:client|project|delivery|transformation)\b'
                ]
            }
            
            industry_scores = {}
            for industry, patterns in industry_indicators.items():
                score = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text, re.IGNORECASE))
                    score += matches
                industry_scores[industry] = score
            
            likely_industry = max(industry_scores.items(), key=lambda x: x[1]) if industry_scores else ('unknown', 0)
            
            return {
                'career_level': likely_level[0],
                'career_level_confidence': likely_level[1],
                'industry': likely_industry[0],
                'industry_confidence': likely_industry[1],
                'career_indicators': career_level_scores,
                'industry_indicators': industry_scores
            }
            
        except Exception as e:
            return {'error': f'Professional profile analysis failed: {str(e)}'}
    
    def _generate_nlp_summary(self, sentiment, emotions, skills, communication) -> Dict:
        """Generate comprehensive NLP summary"""
        try:
            summary = {
                'overall_sentiment': sentiment.get('overall_sentiment', 'Unknown'),
                'dominant_emotion': emotions.get('dominant_emotion', 'Unknown'),
                'communication_quality': communication.get('communication_quality', 'Unknown'),
                'total_skills': skills.get('total_skills', 0),
                'skill_categories': skills.get('skill_categories', []),
                'professional_assessment': self._generate_professional_assessment(sentiment, emotions, skills, communication)
            }
            return summary
        except Exception as e:
            return {'error': f'Summary generation failed: {str(e)}'}
    
    def _generate_professional_assessment(self, sentiment, emotions, skills, communication) -> str:
        """Generate professional assessment based on NLP analysis"""
        try:
            assessment_parts = []
            
            # Sentiment assessment
            if sentiment.get('overall_sentiment') == 'POSITIVE':
                assessment_parts.append("Shows positive attitude and enthusiasm")
            elif sentiment.get('overall_sentiment') == 'NEGATIVE':
                assessment_parts.append("May need to work on maintaining positive tone")
            
            # Communication quality
            comm_quality = communication.get('communication_quality', 'Unknown')
            if comm_quality == 'Excellent':
                assessment_parts.append("Excellent communication skills")
            elif comm_quality == 'Good':
                assessment_parts.append("Good communication skills")
            elif comm_quality == 'Needs Improvement':
                assessment_parts.append("Communication skills need improvement")
            
            # Skills assessment
            total_skills = skills.get('total_skills', 0)
            if total_skills > 10:
                assessment_parts.append("Demonstrates diverse skill set")
            elif total_skills > 5:
                assessment_parts.append("Shows good technical competency")
            else:
                assessment_parts.append("Could benefit from highlighting more specific skills")
            
            return ". ".join(assessment_parts) if assessment_parts else "Assessment pending"
            
        except Exception as e:
            return f"Assessment generation failed: {str(e)}"
