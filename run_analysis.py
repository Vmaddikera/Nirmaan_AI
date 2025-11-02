#!/usr/bin/env python3
"""
Terminal-based AI Candidate Analysis
Run ML pipeline directly without web interface
"""

import sys
import os
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_candidate_analysis():
    """Run complete candidate analysis pipeline"""
    
    print("=" * 60)
    print("AI CANDIDATE ANALYSIS - TERMINAL VERSION")
    print("=" * 60)
    
    try:
        # Import our services
        from services.emotion_analyzer import EmotionAnalyzer
        from services.gaze_analyzer import GazeAnalyzer
        from services.nlp_analyzer import NLPAnalyzer
        from services.transcript_analyzer import TranscriptAnalyzer
        from services.candidate_assessor import CandidateAssessor
        
        print("\n1. INITIALIZING AI MODELS...")
        print("-" * 40)
        
        # Initialize analyzers
        emotion_analyzer = EmotionAnalyzer()
        gaze_analyzer = GazeAnalyzer()
        nlp_analyzer = NLPAnalyzer()
        transcript_analyzer = TranscriptAnalyzer()
        candidate_assessor = CandidateAssessor()
        
        print("SUCCESS: All AI models loaded successfully!")
        
        print("\n2. SAMPLE INPUT DATA")
        print("-" * 40)
        
        # Sample emotion data
        emotion_data = pd.DataFrame([{
            'anger': 0.1,
            'disgust': 0.05,
            'fear': 0.15,
            'joy': 0.6,
            'neutral': 0.2,
            'sadness': 0.05,
            'surprise': 0.1
        }])
        
        # Sample gaze data
        gaze_data = pd.DataFrame([{
            'fixation_duration': 2.5,
            'saccade_amplitude': 15.3,
            'pupil_diameter': 4.2,
            'gaze_x': 512.5,
            'gaze_y': 384.2,
            'blink_rate': 0.8,
            'attention_span': 0.85
        }])
        
        # Sample text data
        transcript_text = "I believe I am a strong candidate for this position because of my extensive experience in software development and my passion for creating innovative solutions. I have worked on multiple projects that required both technical expertise and strong communication skills."
        script_text = "Tell me about yourself and why you are interested in this position."
        
        print("EMOTION SCORES:")
        for emotion, score in emotion_data.iloc[0].items():
            print(f"   {emotion.capitalize()}: {score}")
        
        print("\nGAZE METRICS:")
        for metric, value in gaze_data.iloc[0].items():
            print(f"   {metric.replace('_', ' ').title()}: {value}")
        
        print(f"\nTRANSCRIPT: {transcript_text[:100]}...")
        print(f"SCRIPT: {script_text}")
        
        print("\n3. RUNNING AI ANALYSIS...")
        print("-" * 40)
        
        # Run emotion analysis
        print("Analyzing emotions...")
        emotion_results = emotion_analyzer.analyze(emotion_data)
        if 'error' in emotion_results:
            print(f"   ERROR: {emotion_results['error']}")
        else:
            print(f"   Dominant Emotion: {emotion_results.get('dominant_emotion', 'N/A')}")
            print(f"   Emotion Score: {emotion_results.get('emotion_score', 0):.2f}")
            print(f"   Interpretation: {emotion_results.get('interpretation', 'N/A')}")
            print(f"   Emotion Stability: {emotion_results.get('emotion_stability', 0):.3f}")
            print(f"   Emotion Profile: {emotion_results.get('emotion_profile', {})}")
        
        # Run gaze analysis
        print("\nAnalyzing gaze patterns...")
        gaze_results = gaze_analyzer.analyze(gaze_data)
        if 'error' in gaze_results:
            print(f"   ERROR: {gaze_results['error']}")
        else:
            print(f"   Attention Score: {gaze_results.get('attention_score', 0):.2f}")
            print(f"   Focus Level: {gaze_results.get('focus_level', 'N/A')}")
            print(f"   Eye Contact: {gaze_results.get('eye_contact', 'N/A')}")
            print(f"   Gaze Stability: {gaze_results.get('gaze_stability', 0):.3f}")
            print(f"   Gaze Anomalies: {len(gaze_results.get('anomalies', []))}")
        
        # Run NLP analysis
        print("\nAnalyzing text with AI...")
        nlp_results = nlp_analyzer.analyze(transcript_text, script_text)
        if 'error' in nlp_results:
            print(f"   ERROR: {nlp_results['error']}")
        else:
            print(f"   Sentiment: {nlp_results.get('sentiment', 'N/A')}")
            print(f"   Confidence: {nlp_results.get('sentiment_confidence', 0):.2f}")
            print(f"   Key Skills: {', '.join(nlp_results.get('key_skills', []))}")
            print(f"   Communication Quality: {nlp_results.get('communication', {}).get('communication_quality', 'N/A')}")
            print(f"   Professional Level: {nlp_results.get('profile', {}).get('career_level', 'N/A')}")
            print(f"   Industry: {nlp_results.get('profile', {}).get('industry', 'N/A')}")
        
        # Run transcript analysis
        print("\nAnalyzing transcript...")
        transcript_results = transcript_analyzer.analyze(transcript_text)
        if 'error' in transcript_results:
            print(f"   ERROR: {transcript_results['error']}")
        else:
            print(f"   Communication Score: {transcript_results.get('communication_score', 0):.2f}")
            print(f"   Clarity Score: {transcript_results.get('clarity_score', 0):.2f}")
            print(f"   Response Quality: {transcript_results.get('response_quality', 'N/A')}")
            print(f"   Text Statistics: {transcript_results.get('text_statistics', {})}")
            print(f"   Quality Indicators: {transcript_results.get('quality_indicators', {})}")
        
        # Run final assessment
        print("\nGENERATING FINAL ASSESSMENT...")
        print("-" * 40)
        
        assessment_results = candidate_assessor.assess_candidate(
            emotion_results, gaze_results, nlp_results, transcript_results
        )
        
        if 'error' in assessment_results:
            print(f"   ERROR: {assessment_results['error']}")
        else:
            print("OVERALL RESULTS:")
            print(f"   Overall Score: {assessment_results.get('overall_score', 0):.2f}/10")
            print(f"   Recommendation: {assessment_results.get('recommendation', 'N/A')}")
            
            print("\nCOMPONENT SCORES:")
            component_scores = assessment_results.get('component_scores', {})
            for component, score in component_scores.items():
                print(f"   {component.capitalize()}: {score:.2f}")
            
            print("\nSTRENGTHS:")
            for strength in assessment_results.get('strengths', []):
                print(f"   - {strength}")
            
            print("\nAREAS FOR IMPROVEMENT:")
            for weakness in assessment_results.get('weaknesses', []):
                print(f"   - {weakness}")
            
            print("\nKEY INSIGHTS:")
            for insight in assessment_results.get('insights', []):
                print(f"   - {insight}")
            
            print("\nDETAILED ASSESSMENT:")
            detailed = assessment_results.get('detailed_assessment', {})
            for component, details in detailed.items():
                print(f"   {component.capitalize()}: {details.get('assessment', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    run_candidate_analysis()
