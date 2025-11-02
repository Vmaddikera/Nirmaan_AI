#!/usr/bin/env python3


from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from Script.emotion_analyzer import EmotionAnalyzer
from Script.gaze_analyzer import GazeAnalyzer
from Script.nlp_analyzer import NLPAnalyzer
from Script.transcript_analyzer import TranscriptAnalyzer
from Script.candidate_assessor import CandidateAssessor
from utils.data_validator import DataValidator
from utils.logger import get_logger
from utils.exceptions import *
from datetime import datetime
import traceback

app = Flask(__name__)

# Initialize logger
logger = get_logger()

# Initialize analyzers with error handling
try:
    logger.log_system_status("Initializing analyzers")
    emotion_analyzer = EmotionAnalyzer()
    gaze_analyzer = GazeAnalyzer()
    nlp_analyzer = NLPAnalyzer()
    transcript_analyzer = TranscriptAnalyzer()
    candidate_assessor = CandidateAssessor()
    data_validator = DataValidator()
    logger.log_system_status("All analyzers initialized successfully")
except Exception as e:
    logger.log_analysis_error("SystemInitialization", e)
    raise SystemError(f"Failed to initialize analysis system: {str(e)}")

def _clean_nan_values(data):
    """Clean NaN values and convert NumPy types to JSON-serializable types"""
    import math
    import numpy as np
    
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            # Handle NumPy types and convert to Python native types
            if isinstance(value, (np.integer, np.floating)):
                if np.isnan(value) or np.isinf(value):
                    cleaned[key] = 0.0
                else:
                    cleaned[key] = float(value)  # Convert to Python float
            elif isinstance(value, (int, float)):
                if math.isnan(value) or value is None or (isinstance(value, float) and np.isnan(value)):
                    cleaned[key] = 0.0
                elif value == float('inf') or value == float('-inf'):
                    cleaned[key] = 0.0
                else:
                    cleaned[key] = value
            elif isinstance(value, np.ndarray):
                # Convert NumPy arrays to Python lists
                cleaned[key] = value.tolist()
            elif isinstance(value, list):
                cleaned[key] = [_clean_nan_values(item) if isinstance(item, dict) else item for item in value]
            elif isinstance(value, dict):
                cleaned[key] = _clean_nan_values(value)
            else:
                cleaned[key] = value
        return cleaned
    elif isinstance(data, list):
        return [_clean_nan_values(item) if isinstance(item, dict) else item for item in data]
    elif isinstance(data, (np.integer, np.floating)):
        if np.isnan(data) or np.isinf(data):
            return 0.0
        else:
            return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    return data

@app.route('/')
def index():
    """Main page with input form"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_candidate():
    """Analyze candidate data and return results"""
    start_time = datetime.now()
    
    try:
        logger.log_user_action("Analysis request started", {
            'endpoint': '/api/analyze',
            'method': 'POST',
            'timestamp': start_time.isoformat()
        })
        
        # Get form data
        data = request.form.to_dict()
        logger.logger.debug(f"Received form data with {len(data)} fields")
        
        # Convert form data to the format expected by validator
        form_data = {
            'anger': data.get('anger', '0'),
            'disgust': data.get('disgust', '0'),
            'fear': data.get('fear', '0'),
            'joy': data.get('joy', '0'),
            'neutral': data.get('neutral', '0'),
            'sadness': data.get('sadness', '0'),
            'surprise': data.get('surprise', '0'),
            'fixation_duration': data.get('fixation_duration', '0'),
            'saccade_amplitude': data.get('saccade_amplitude', '0'),
            'pupil_diameter': data.get('pupil_diameter', '0'),
            'gaze_x': data.get('gaze_x', '0'),
            'gaze_y': data.get('gaze_y', '0'),
            'blink_rate': data.get('blink_rate', '0'),
            'attention_span': data.get('attention_span', '0'),
            'transcript_text': data.get('transcript_text', ''),
            'script_text': data.get('script_text', '')
        }
        
        logger.logger.info(f"Processing form data with {len(form_data)} fields")
        
        # Process form input - convert form data to proper structure
        try:
            logger.logger.debug("Converting form data to features")
            
            emotion_features = {
                'anger': float(data.get('anger', 0)),
                'disgust': float(data.get('disgust', 0)),
                'fear': float(data.get('fear', 0)),
                'joy': float(data.get('joy', 0)),
                'neutral': float(data.get('neutral', 0)),
                'sadness': float(data.get('sadness', 0)),
                'surprise': float(data.get('surprise', 0))
            }
            logger.logger.debug(f"Emotion features: {emotion_features}")
            
            gaze_features = {
                'fixation_duration': float(data.get('fixation_duration', 0)),
                'saccade_amplitude': float(data.get('saccade_amplitude', 0)),
                'pupil_diameter': float(data.get('pupil_diameter', 0)),
                'gaze_x': float(data.get('gaze_x', 0)),
                'gaze_y': float(data.get('gaze_y', 0)),
                'blink_rate': float(data.get('blink_rate', 0)),
                'attention_span': float(data.get('attention_span', 0))
            }
            logger.logger.debug(f"Gaze features: {gaze_features}")
            
            emotion_data = pd.DataFrame([emotion_features])
            gaze_data = pd.DataFrame([gaze_features])
            transcript_text = data.get('transcript_text', '')
            
            logger.logger.debug(f"Text data - transcript: {len(transcript_text)} chars")
            
        except ValueError as e:
            logger.log_analysis_error("DataConversion", e, {'step': 'numeric_conversion'})
            raise DataProcessingError(f'Invalid numeric values: {str(e)}', 'numeric_conversion')
        except Exception as e:
            logger.log_analysis_error("DataConversion", e, {'step': 'data_conversion'})
            raise DataProcessingError(f'Data conversion failed: {str(e)}', 'data_conversion')
        
        # Run analysis pipeline
        results = {}
        
        try:
            # 1. Emotion Analysis
            emotion_start = datetime.now()
            logger.log_analysis_start("EmotionAnalysis", {'data_shape': emotion_data.shape})
            
            try:
                emotion_results = emotion_analyzer.analyze(emotion_data)
                emotion_results = _clean_nan_values(emotion_results)
                results['emotion'] = emotion_results
                logger.log_analysis_success("EmotionAnalysis", {'dominant_emotion': emotion_results.get('dominant_emotion')})
                logger.log_performance("EmotionAnalysis", emotion_start, datetime.now())
            except Exception as e:
                logger.log_analysis_error("EmotionAnalysis", e, {'data_shape': emotion_data.shape})
                raise EmotionAnalysisError(f"Emotion analysis failed: {str(e)}", {'data_shape': emotion_data.shape})
            
            # 2. Gaze Analysis
            gaze_start = datetime.now()
            logger.log_analysis_start("GazeAnalysis", {'data_shape': gaze_data.shape})
            
            try:
                gaze_results = gaze_analyzer.analyze(gaze_data)
                gaze_results = _clean_nan_values(gaze_results)
                results['gaze'] = gaze_results
                logger.log_analysis_success("GazeAnalysis", {'attention_score': gaze_results.get('attention_score')})
                logger.log_performance("GazeAnalysis", gaze_start, datetime.now())
            except Exception as e:
                logger.log_analysis_error("GazeAnalysis", e, {'data_shape': gaze_data.shape})
                raise GazeAnalysisError(f"Gaze analysis failed: {str(e)}", {'data_shape': gaze_data.shape})
            
            # 3. NLP Analysis
            nlp_start = datetime.now()
            logger.log_analysis_start("NLPAnalysis", {'transcript_length': len(transcript_text)})
            
            try:
                nlp_results = nlp_analyzer.analyze(transcript_text)
                nlp_results = _clean_nan_values(nlp_results)
                results['nlp'] = nlp_results
                logger.log_analysis_success("NLPAnalysis", {'sentiment': nlp_results.get('sentiment')})
                logger.log_performance("NLPAnalysis", nlp_start, datetime.now())
            except Exception as e:
                logger.log_analysis_error("NLPAnalysis", e, {'transcript_length': len(transcript_text)})
                raise NLPAnalysisError(f"NLP analysis failed: {str(e)}", {'transcript_length': len(transcript_text)})
            
            # 4. Transcript Analysis
            transcript_start = datetime.now()
            logger.log_analysis_start("TranscriptAnalysis", {'transcript_length': len(transcript_text)})
            
            try:
                transcript_results = transcript_analyzer.analyze(transcript_text)
                transcript_results = _clean_nan_values(transcript_results)
                results['transcript'] = transcript_results
                logger.log_analysis_success("TranscriptAnalysis", {'communication_score': transcript_results.get('communication_score')})
                logger.log_performance("TranscriptAnalysis", transcript_start, datetime.now())
            except Exception as e:
                logger.log_analysis_error("TranscriptAnalysis", e, {'transcript_length': len(transcript_text)})
                raise TranscriptAnalysisError(f"Transcript analysis failed: {str(e)}", {'transcript_length': len(transcript_text)})
            
            # 5. Final Assessment
            assessment_start = datetime.now()
            logger.log_analysis_start("FinalAssessment", {'components': list(results.keys())})
            
            try:
                assessment_results = candidate_assessor.assess_candidate(
                    emotion_results, gaze_results, nlp_results, transcript_results, transcript_text
                )
                assessment_results = _clean_nan_values(assessment_results)
                results['assessment'] = assessment_results
                logger.log_analysis_success("FinalAssessment", {'overall_score': assessment_results.get('overall_score')})
                logger.log_performance("FinalAssessment", assessment_start, datetime.now())
            except Exception as e:
                logger.log_analysis_error("FinalAssessment", e, {'components': list(results.keys())})
                raise AssessmentError(f"Final assessment failed: {str(e)}", {'components': list(results.keys())})
            
        except (EmotionAnalysisError, GazeAnalysisError, NLPAnalysisError, TranscriptAnalysisError, AssessmentError) as e:
            logger.logger.error(f"Analysis pipeline failed: {str(e)}")
            return jsonify({'error': f'Analysis failed: {str(e)}', 'component': e.component}), 500
        except Exception as e:
            logger.log_analysis_error("AnalysisPipeline", e, {'step': 'unknown'})
            return jsonify({'error': f'Unexpected analysis error: {str(e)}'}), 500
        
        # Test JSON serialization before returning
        try:
            import json
            # Clean results again to ensure all NumPy types are converted
            cleaned_results = _clean_nan_values(results)
            json.dumps(cleaned_results)
            logger.logger.debug("JSON serialization test passed")
            # Use cleaned results for response
            results = cleaned_results
        except Exception as e:
            logger.log_analysis_error("JSONSerialization", e, {'results_keys': list(results.keys())})
            raise JSONSerializationError(f'Results contain invalid JSON values: {str(e)}', 'analysis_results')
        
        # Log successful completion
        total_time = (datetime.now() - start_time).total_seconds()
        logger.log_performance("CompleteAnalysis", start_time, datetime.now())
        logger.log_user_action("Analysis completed successfully", {
            'total_time': total_time,
            'components_analyzed': len(results),
            'overall_score': results.get('assessment', {}).get('overall_score', 'N/A')
        })
        
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'processing_time': total_time
        })
        
    except (DataProcessingError, EmotionAnalysisError, GazeAnalysisError, NLPAnalysisError, 
            TranscriptAnalysisError, AssessmentError, JSONSerializationError) as e:
        logger.logger.error(f"Analysis failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Analysis failed: {str(e)}',
            'component': e.component,
            'error_code': e.error_code
        }), 500
    except Exception as e:
        logger.log_analysis_error("UnexpectedError", e, {'endpoint': '/api/analyze'})
        return jsonify({
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'emotion_analyzer': 'ready',
            'gaze_analyzer': 'ready',
            'nlp_analyzer': 'ready',
            'transcript_analyzer': 'ready',
            'candidate_assessor': 'ready'
        }
    })


if __name__ == '__main__':
    print("Starting AI Candidate Analysis Web Application...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
