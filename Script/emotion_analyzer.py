import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tabulate import tabulate
import json
from typing import Dict, List, Tuple

class EmotionAnalyzer:
    def __init__(self):
        self.emotion_columns = [
            'anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise'
        ]
    
    def analyze(self, emotion_data):
        """
        Analyze emotion data and return comprehensive results
        """
        try:
            # Ensure we have the required emotion columns
            available_columns = [col for col in self.emotion_columns if col in emotion_data.columns]
            
            if not available_columns:
                return {'error': 'No emotion columns found in data'}
            
            # Calculate emotion statistics
            emotion_stats = self._calculate_emotion_statistics(emotion_data[available_columns])
            
            # Find dominant emotion
            dominant_emotion_tuple = self._find_dominant_emotion(emotion_stats)
            dominant_emotion = dominant_emotion_tuple[0]
            
            # Calculate emotion stability (variance)
            emotion_stability = self._calculate_emotion_stability(emotion_data[available_columns])
            
            # Generate emotion profile
            emotion_profile = self._generate_emotion_profile(emotion_stats, dominant_emotion_tuple)
            
            # Calculate overall emotion score
            emotion_score = np.mean(list(emotion_stats.values()))
            
            return {
                'emotion_scores': emotion_stats,
                'dominant_emotion': dominant_emotion,
                'emotion_score': emotion_score,
                'emotion_stability': emotion_stability,
                'emotion_profile': emotion_profile,
                'interpretation': 'Positive' if emotion_score > 0.5 else 'Neutral' if emotion_score > 0.3 else 'Negative',
                'summary': self._generate_emotion_summary(emotion_stats, dominant_emotion_tuple, emotion_stability)
            }
            
        except Exception as e:
            return {'error': f'Emotion analysis failed: {str(e)}'}
    
    def _calculate_emotion_statistics(self, emotion_data):
        """Calculate mean emotion scores"""
        return emotion_data.mean().to_dict()
    
    def _find_dominant_emotion(self, emotion_stats):
        """Find the emotion with highest average score"""
        return max(emotion_stats.items(), key=lambda x: x[1])
    
    def _calculate_emotion_stability(self, emotion_data):
        """Calculate emotion stability (lower variance = more stable)"""
        return emotion_data.var().mean()
    
    def _generate_emotion_profile(self, emotion_stats, dominant_emotion):
        """Generate detailed emotion profile"""
        profile = {
            'primary_emotion': dominant_emotion[0],
            'primary_score': dominant_emotion[1],
            'emotion_distribution': emotion_stats,
            'emotional_range': max(emotion_stats.values()) - min(emotion_stats.values()),
            'emotional_balance': self._calculate_emotional_balance(emotion_stats)
        }
        return profile
    
    def _calculate_emotional_balance(self, emotion_stats):
        """Calculate how balanced the emotions are"""
        values = list(emotion_stats.values())
        mean_val = np.mean(values)
        if mean_val > 0:
            std_val = np.std(values)
            balance = 1 - (std_val / mean_val)
            return max(0, min(1, balance))  # Ensure between 0 and 1
        else:
            return 0.0
    
    def _generate_emotion_summary(self, emotion_stats, dominant_emotion, stability):
        """Generate human-readable emotion summary"""
        primary_emotion = dominant_emotion[0]
        primary_score = dominant_emotion[1]
        
        # Determine emotional stability level
        if stability < 0.01:
            stability_level = "Very Stable"
        elif stability < 0.05:
            stability_level = "Stable"
        elif stability < 0.1:
            stability_level = "Moderately Variable"
        else:
            stability_level = "Highly Variable"
        
        # Determine emotional intensity
        if primary_score > 0.7:
            intensity_level = "High"
        elif primary_score > 0.4:
            intensity_level = "Moderate"
        else:
            intensity_level = "Low"
        
        return {
            'dominant_emotion': primary_emotion,
            'intensity': intensity_level,
            'stability': stability_level,
            'description': f"Candidate shows {intensity_level.lower()} {primary_emotion} with {stability_level.lower()} emotional patterns"
        }
