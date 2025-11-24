from flask import Flask, request, jsonify, render_template
from Script.rubric_transcript_analyzer import RubricTranscriptAnalyzer
from utils.logger import log_info, log_error, log_analysis
from utils.exceptions import TranscriptError, ValidationError
from datetime import datetime
import traceback

app = Flask(__name__)

# Initialize analyzer
print("="*80)
print("INITIALIZING RUBRIC-BASED TRANSCRIPT ANALYZER")
print("="*80)
try:
    analyzer = RubricTranscriptAnalyzer()
    log_info("Analyzer initialized successfully")
    print("✓ Analyzer initialized successfully!\n")
except Exception as e:
    log_error("Failed to initialize analyzer", e)
    print(f"✗ Failed to initialize: {str(e)}")
    raise

@app.route('/')
def index():
    """Main page with input form"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_transcript():
    """Analyze transcript and return results"""
    start_time = datetime.now()
    
    try:
        # Get form data or JSON data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        # Extract transcript and optional duration
        transcript_text = data.get('transcript_text', '')
        duration_seconds = data.get('duration_seconds')
        
        # Validate input
        if not transcript_text or not transcript_text.strip():
            log_error("Empty transcript received")
            raise ValidationError('Transcript text is required')
        
        # Convert duration to int if provided
        if duration_seconds:
            try:
                duration_seconds = int(duration_seconds)
            except (ValueError, TypeError):
                duration_seconds = None
        
        log_info(f"Starting analysis | Length: {len(transcript_text)} chars | Duration: {duration_seconds}s")
        
        print(f"\n{'='*80}")
        print(f"ANALYZING TRANSCRIPT")
        print(f"{'='*80}")
        print(f"Length: {len(transcript_text)} characters")
        print(f"Duration: {duration_seconds} seconds" if duration_seconds else "Duration: Not provided")
        print(f"{'='*80}\n")
        
        # Run analysis
        try:
            results = analyzer.analyze(transcript_text, duration_seconds)
        except Exception as e:
            log_error("Analysis failed", e)
            print(f"✗ Analysis failed: {str(e)}")
            traceback.print_exc()
            raise TranscriptError(f'Analysis failed: {str(e)}')
        
        # Check if analysis returned an error
        if 'error' in results:
            return jsonify({
                'success': False,
                'error': results['error']
            }), 500
        
        # Calculate processing time
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Log successful analysis
        log_analysis(len(transcript_text), total_time, results['overall_score'])
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"Overall Score: {results['overall_score']}/{results['max_possible_score']}")
        print(f"Percentage: {results['percentage']}%")
        print(f"Performance Level: {results['performance_level']}")
        print(f"Processing Time: {total_time:.2f}s")
        print(f"{'='*80}\n")
        
        # Return results
        return jsonify({
            'success': True,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(total_time, 2)
        })
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': str(e.message)
        }), 400
    except TranscriptError as e:
        return jsonify({
            'success': False,
            'error': str(e.message)
        }), 500
    except Exception as e:
        log_error("Unexpected error in analyze endpoint", e)
        print(f"✗ Unexpected error: {str(e)}")
        traceback.print_exc()
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
        'analyzer': 'ready',
        'version': '2.0 - Rubric Based'
    })


if __name__ == '__main__':
    import os
    
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Check if running in production
    is_production = os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('FLASK_ENV') == 'production'
    
    print("\n" + "="*80)
    print("RUBRIC-BASED TRANSCRIPT ANALYSIS SYSTEM")
    print("="*80)
    print(f"Environment: {'Production' if is_production else 'Development'}")
    print(f"Server starting on port {port}...")
    if not is_production:
        print("Open your browser and go to: http://localhost:5000")
    print("="*80 + "\n")
    
    app.run(debug=not is_production, host='0.0.0.0', port=port)
