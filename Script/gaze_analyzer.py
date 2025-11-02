import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, mannwhitneyu, kruskal
import json
from typing import Dict, List, Tuple

class GazeAnalyzer:
    def __init__(self):
        self.gaze_metrics = [
            'fixation_duration', 'saccade_amplitude', 'pupil_diameter',
            'gaze_x', 'gaze_y', 'blink_rate', 'attention_span'
        ]
    
    def analyze(self, gaze_data):
        """
        Analyze gaze data and return comprehensive results
        """
        try:
            # Get available gaze columns
            available_columns = [col for col in self.gaze_metrics if col in gaze_data.columns]
            
            if not available_columns:
                return {'error': 'No gaze metrics found in data'}
            
            # Calculate gaze statistics
            gaze_stats = self._calculate_gaze_statistics(gaze_data[available_columns])
            
            # Analyze attention patterns
            attention_analysis = self._analyze_attention_patterns(gaze_data[available_columns])
            
            # Calculate gaze stability
            gaze_stability = self._calculate_gaze_stability(gaze_data[available_columns])
            
            # Detect gaze anomalies
            anomalies = self._detect_gaze_anomalies(gaze_data[available_columns])
            
            # Generate gaze profile
            gaze_profile = self._generate_gaze_profile(gaze_stats, attention_analysis, gaze_stability)
            
            # Calculate attention score and focus level
            attention_score = gaze_data['attention_span'].iloc[0] if 'attention_span' in gaze_data.columns else 0.5
            focus_level = "High" if attention_score > 0.7 else "Medium" if attention_score > 0.4 else "Low"
            eye_contact = "Good" if gaze_data['blink_rate'].iloc[0] < 1.0 else "Needs Improvement" if 'blink_rate' in gaze_data.columns else "Unknown"
            
            return {
                'gaze_statistics': gaze_stats,
                'attention_analysis': attention_analysis,
                'attention_score': attention_score,
                'focus_level': focus_level,
                'eye_contact': eye_contact,
                'gaze_stability': gaze_stability,
                'anomalies': anomalies,
                'gaze_profile': gaze_profile,
                'summary': self._generate_gaze_summary(gaze_stats, attention_analysis, gaze_stability)
            }
            
        except Exception as e:
            return {'error': f'Gaze analysis failed: {str(e)}'}
    
    def _calculate_gaze_statistics(self, gaze_data):
        """Calculate comprehensive gaze statistics"""
        stats_dict = {}
        
        for column in gaze_data.columns:
            if gaze_data[column].dtype in ['float64', 'int64']:
                stats_dict[column] = {
                    'mean': float(gaze_data[column].mean()),
                    'std': float(gaze_data[column].std()),
                    'min': float(gaze_data[column].min()),
                    'max': float(gaze_data[column].max()),
                    'median': float(gaze_data[column].median()),
                    'q25': float(gaze_data[column].quantile(0.25)),
                    'q75': float(gaze_data[column].quantile(0.75))
                }
        
        return stats_dict
    
    def _analyze_attention_patterns(self, gaze_data):
        """Analyze attention and focus patterns"""
        analysis = {}
        
        # Calculate attention span if available
        if 'attention_span' in gaze_data.columns:
            attention_span = gaze_data['attention_span']
            analysis['average_attention_span'] = float(attention_span.mean())
            analysis['attention_consistency'] = float(1 - attention_span.std() / attention_span.mean()) if attention_span.mean() > 0 else 0
        
        # Analyze fixation patterns
        if 'fixation_duration' in gaze_data.columns:
            fixation_duration = gaze_data['fixation_duration']
            analysis['average_fixation_duration'] = float(fixation_duration.mean())
            analysis['fixation_stability'] = float(1 - fixation_duration.std() / fixation_duration.mean()) if fixation_duration.mean() > 0 else 0
        
        # Analyze saccade patterns
        if 'saccade_amplitude' in gaze_data.columns:
            saccade_amplitude = gaze_data['saccade_amplitude']
            analysis['average_saccade_amplitude'] = float(saccade_amplitude.mean())
            analysis['saccade_consistency'] = float(1 - saccade_amplitude.std() / saccade_amplitude.mean()) if saccade_amplitude.mean() > 0 else 0
        
        # Analyze pupil response
        if 'pupil_diameter' in gaze_data.columns:
            pupil_diameter = gaze_data['pupil_diameter']
            analysis['average_pupil_diameter'] = float(pupil_diameter.mean())
            analysis['pupil_variability'] = float(pupil_diameter.std())
        
        return analysis
    
    def _calculate_gaze_stability(self, gaze_data):
        """Calculate gaze stability metrics"""
        stability_metrics = {}
        
        for column in gaze_data.columns:
            if gaze_data[column].dtype in ['float64', 'int64']:
                # Calculate coefficient of variation (lower = more stable)
                mean_val = gaze_data[column].mean()
                std_val = gaze_data[column].std()
                if mean_val > 0:
                    cv = std_val / mean_val
                else:
                    cv = 0.0  # Use 0 instead of inf
                
                stability_metrics[column] = {
                    'coefficient_of_variation': float(cv),
                    'stability_score': float(1 / (1 + cv))
                }
        
        return stability_metrics
    
    def _detect_gaze_anomalies(self, gaze_data):
        """Detect unusual gaze patterns"""
        anomalies = {}
        
        for column in gaze_data.columns:
            if gaze_data[column].dtype in ['float64', 'int64']:
                # Use IQR method to detect outliers
                Q1 = gaze_data[column].quantile(0.25)
                Q3 = gaze_data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = gaze_data[(gaze_data[column] < lower_bound) | (gaze_data[column] > upper_bound)]
                anomaly_count = len(outliers)
                anomaly_percentage = (anomaly_count / len(gaze_data)) * 100
                
                anomalies[column] = {
                    'outlier_count': anomaly_count,
                    'outlier_percentage': float(anomaly_percentage),
                    'is_anomalous': anomaly_percentage > 5  # Flag if more than 5% outliers
                }
        
        return anomalies
    
    def _generate_gaze_profile(self, gaze_stats, attention_analysis, gaze_stability):
        """Generate comprehensive gaze profile"""
        profile = {
            'focus_level': self._assess_focus_level(attention_analysis),
            'attention_quality': self._assess_attention_quality(attention_analysis),
            'gaze_consistency': self._assess_gaze_consistency(gaze_stability),
            'overall_engagement': self._assess_overall_engagement(gaze_stats, attention_analysis)
        }
        return profile
    
    def _assess_focus_level(self, attention_analysis):
        """Assess focus level based on attention patterns"""
        if 'average_attention_span' in attention_analysis:
            span = attention_analysis['average_attention_span']
            if span > 0.8:
                return "High Focus"
            elif span > 0.5:
                return "Moderate Focus"
            else:
                return "Low Focus"
        return "Unknown"
    
    def _assess_attention_quality(self, attention_analysis):
        """Assess quality of attention"""
        if 'attention_consistency' in attention_analysis:
            consistency = attention_analysis['attention_consistency']
            if consistency > 0.8:
                return "Consistent"
            elif consistency > 0.5:
                return "Moderately Consistent"
            else:
                return "Inconsistent"
        return "Unknown"
    
    def _assess_gaze_consistency(self, gaze_stability):
        """Assess overall gaze consistency"""
        if not gaze_stability:
            return "Unknown"
        
        avg_stability = np.mean([metrics['stability_score'] for metrics in gaze_stability.values()])
        if avg_stability > 0.7:
            return "Very Consistent"
        elif avg_stability > 0.5:
            return "Moderately Consistent"
        else:
            return "Inconsistent"
    
    def _assess_overall_engagement(self, gaze_stats, attention_analysis):
        """Assess overall engagement level"""
        # Combine multiple metrics to assess engagement
        engagement_score = 0
        metrics_count = 0
        
        # Consider attention span
        if 'average_attention_span' in attention_analysis:
            engagement_score += attention_analysis['average_attention_span']
            metrics_count += 1
        
        # Consider fixation duration
        if 'average_fixation_duration' in attention_analysis:
            # Normalize fixation duration (assuming longer is better)
            fixation_score = min(attention_analysis['average_fixation_duration'] / 2.0, 1.0)
            engagement_score += fixation_score
            metrics_count += 1
        
        if metrics_count > 0:
            avg_engagement = engagement_score / metrics_count
            if avg_engagement > 0.7:
                return "High Engagement"
            elif avg_engagement > 0.4:
                return "Moderate Engagement"
            else:
                return "Low Engagement"
        
        return "Unknown"
    
    def _generate_gaze_summary(self, gaze_stats, attention_analysis, gaze_stability):
        """Generate human-readable gaze summary"""
        focus_level = self._assess_focus_level(attention_analysis)
        attention_quality = self._assess_attention_quality(attention_analysis)
        gaze_consistency = self._assess_gaze_consistency(gaze_stability)
        engagement = self._assess_overall_engagement(gaze_stats, attention_analysis)
        
        return {
            'focus_level': focus_level,
            'attention_quality': attention_quality,
            'gaze_consistency': gaze_consistency,
            'engagement': engagement,
            'description': f"Candidate shows {focus_level.lower()} with {attention_quality.lower()} attention and {gaze_consistency.lower()} gaze patterns, indicating {engagement.lower()}."
        }
