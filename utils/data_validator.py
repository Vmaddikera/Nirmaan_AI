import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json

class DataValidator:
    def __init__(self):
        self.required_emotion_columns = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
        self.required_gaze_columns = ['fixation_duration', 'saccade_amplitude', 'pupil_diameter', 'gaze_x', 'gaze_y']
        self.required_metadata_columns = []  # No metadata required
    
    def validate_input(self, data: Dict) -> Dict:
        """
        Validate input data structure and content
        """
        try:
            validation_result = {
                'valid': True,
                'message': 'Validation successful',
                'warnings': [],
                'errors': []
            }
            
            # Only validate form input (no JSON input type)
            validation_result = self._validate_form_input(data, validation_result)
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'message': f'Validation failed: {str(e)}',
                'warnings': [],
                'errors': [str(e)]
            }
    
    def _validate_form_input(self, data: Dict, validation_result: Dict) -> Dict:
        """Validate form input format"""
        try:
            # Check required emotion fields
            emotion_fields = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            for field in emotion_fields:
                if field not in data:
                    validation_result['errors'].append(f'Missing emotion field: {field}')
                    validation_result['valid'] = False
                else:
                    try:
                        float(data[field])
                    except (ValueError, TypeError):
                        validation_result['errors'].append(f'Invalid emotion value for {field}: {data[field]}')
                        validation_result['valid'] = False
            
            # Check required gaze fields
            gaze_fields = ['fixation_duration', 'saccade_amplitude', 'pupil_diameter', 
                          'gaze_x', 'gaze_y', 'blink_rate', 'attention_span']
            for field in gaze_fields:
                if field not in data:
                    validation_result['errors'].append(f'Missing gaze field: {field}')
                    validation_result['valid'] = False
                else:
                    try:
                        float(data[field])
                    except (ValueError, TypeError):
                        validation_result['errors'].append(f'Invalid gaze value for {field}: {data[field]}')
                        validation_result['valid'] = False
            
            # Check text fields
            if 'transcript_text' not in data or not data['transcript_text'].strip():
                validation_result['errors'].append('Missing or empty transcript_text')
                validation_result['valid'] = False
            
            if 'script_text' not in data or not data['script_text'].strip():
                validation_result['errors'].append('Missing or empty script_text')
                validation_result['valid'] = False
            
            if validation_result['valid']:
                validation_result['message'] = 'Form input validation successful'
            else:
                validation_result['message'] = 'Form input validation failed'
            
            return validation_result
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['message'] = f'Form validation error: {str(e)}'
            validation_result['errors'].append(str(e))
            return validation_result
    
    # Removed CSV input validation - no longer needed
    
    # Removed JSON input validation - no longer needed
    
    # Removed CSV emotion data validation - no longer needed
    
    # Removed CSV gaze data validation - no longer needed
    
    # Removed metadata validation - no longer needed
    
    def _validate_emotion_features(self, emotion_features: Dict) -> Dict:
        """Validate emotion features for single JSON input"""
        try:
            if not isinstance(emotion_features, dict):
                return {'valid': False, 'errors': ['Emotion features must be a dictionary']}
            
            # Check for at least some emotion columns
            emotion_columns_found = [col for col in self.required_emotion_columns if col in emotion_features]
            if len(emotion_columns_found) < 3:
                return {'valid': False, 'errors': ['Must contain at least 3 emotion features']}
            
            # Validate numeric values
            for col in emotion_columns_found:
                try:
                    float(emotion_features[col])
                except (ValueError, TypeError):
                    return {'valid': False, 'errors': [f'Emotion feature {col} must be numeric']}
            
            return {'valid': True, 'errors': []}
            
        except Exception as e:
            return {'valid': False, 'errors': [f'Emotion features validation error: {str(e)}']}
    
    def _validate_gaze_features(self, gaze_features: Dict) -> Dict:
        """Validate gaze features for single JSON input"""
        try:
            if not isinstance(gaze_features, dict):
                return {'valid': False, 'errors': ['Gaze features must be a dictionary']}
            
            # Check for at least some gaze columns
            gaze_columns_found = [col for col in self.required_gaze_columns if col in gaze_features]
            if len(gaze_columns_found) < 2:
                return {'valid': False, 'errors': ['Must contain at least 2 gaze features']}
            
            # Validate numeric values
            for col in gaze_columns_found:
                try:
                    float(gaze_features[col])
                except (ValueError, TypeError):
                    return {'valid': False, 'errors': [f'Gaze feature {col} must be numeric']}
            
            return {'valid': True, 'errors': []}
            
        except Exception as e:
            return {'valid': False, 'errors': [f'Gaze features validation error: {str(e)}']}
    
    def _validate_metadata_features(self, metadata: Dict) -> Dict:
        """Validate metadata features for single JSON input - now optional"""
        try:
            # Metadata is now optional - always return valid
            return {'valid': True, 'errors': []}
            
        except Exception as e:
            return {'valid': False, 'errors': [f'Metadata validation error: {str(e)}']}
    
    def _validate_text_fields(self, data: Dict) -> Dict:
        """Validate text fields"""
        try:
            text_fields = ['transcript_text', 'script_text']
            errors = []
            
            for field in text_fields:
                if field not in data:
                    errors.append(f'Missing required text field: {field}')
                elif not isinstance(data[field], str):
                    errors.append(f'Text field {field} must be a string')
                elif len(data[field].strip()) < 10:
                    errors.append(f'Text field {field} is too short (minimum 10 characters)')
            
            return {
                'valid': len(errors) == 0,
                'errors': errors
            }
            
        except Exception as e:
            return {'valid': False, 'errors': [f'Text fields validation error: {str(e)}']}
    
    def validate_data_consistency(self, emotion_data: List[Dict], gaze_data: List[Dict], 
                                 metadata: List[Dict]) -> Dict:
        """Validate consistency between different data sources"""
        try:
            validation_result = {
                'valid': True,
                'message': 'Data consistency validation successful',
                'warnings': [],
                'errors': []
            }
            
            # Check if all datasets have the same number of rows
            emotion_count = len(emotion_data)
            gaze_count = len(gaze_data)
            metadata_count = len(metadata)
            
            if emotion_count != gaze_count:
                validation_result['warnings'].append(
                    f'Emotion data has {emotion_count} rows, gaze data has {gaze_count} rows'
                )
            
            if emotion_count != metadata_count:
                validation_result['warnings'].append(
                    f'Emotion data has {emotion_count} rows, metadata has {metadata_count} rows'
                )
            
            # Check for candidate ID consistency if available
            if metadata_count > 0 and 'candidate_id' in metadata[0]:
                candidate_ids = [row.get('candidate_id') for row in metadata]
                unique_candidates = set(candidate_ids)
                
                if len(unique_candidates) > 1:
                    validation_result['warnings'].append(
                        f'Multiple candidates detected: {len(unique_candidates)} unique candidates'
                    )
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'message': f'Data consistency validation failed: {str(e)}',
                'warnings': [],
                'errors': [str(e)]
            }
