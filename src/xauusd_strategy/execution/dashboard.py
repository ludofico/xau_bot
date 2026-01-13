
from flask import Flask, render_template, jsonify, request
import json
import os
import time

app = Flask(__name__)

MONITOR_FILE = "monitor/state.json"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    try:
        if os.path.exists(MONITOR_FILE):
            with open(MONITOR_FILE, 'r') as f:
                data = json.load(f)
            return jsonify(data)
        else:
            return jsonify({'error': 'State file not found', 'timestamp': time.time()})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    # Create a stop signal file
    with open("monitor/stop.signal", 'w') as f:
        f.write("STOP")
    return jsonify({'status': 'STOP SIGNAL SENT'})

if __name__ == '__main__':
    print("Starting Dashboard on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
