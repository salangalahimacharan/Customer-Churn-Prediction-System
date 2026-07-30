"""
app.py - Production Flask Web Application for Customer Churn Prediction

Provides a modern, interactive web dashboard interface and REST API for 
real-time customer churn risk evaluation and retention strategy recommendation.
"""

import os
import sys
from flask import Flask, render_template_string, request, jsonify

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from predict import predict_customer_churn

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.1);
            --accent-blue: #38bdf8;
            --accent-green: #22c55e;
            --accent-red: #ef4444;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem 1rem;
        }

        .container {
            max-width: 1100px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 1.1rem;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }

        @media (max-width: 868px) {
            .grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            border-radius: 1rem;
            padding: 2rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
        }

        .card h2 {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--accent-blue);
            border-bottom: 1px solid var(--card-border);
            padding-bottom: 0.5rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.2rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }

        .form-group.full {
            grid-column: span 2;
        }

        label {
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 500;
        }

        input, select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 0.5rem;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.2s;
        }

        input:focus, select:focus {
            border-color: var(--accent-blue);
        }

        button {
            background: linear-gradient(to right, #0284c7, #6366f1);
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 1rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 1.5rem;
            transition: opacity 0.2s, transform 0.1s;
        }

        button:hover {
            opacity: 0.95;
        }

        button:active {
            transform: scale(0.99);
        }

        .result-box {
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .status-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 9999px;
            font-weight: 700;
            font-size: 1.1rem;
            text-align: center;
        }

        .badge-churn {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }

        .badge-retain {
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.4);
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(15, 23, 42, 0.4);
            padding: 1rem;
            border-radius: 0.5rem;
        }

        .metric-label {
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        .metric-val {
            font-weight: 700;
            font-size: 1.1rem;
        }

        .progress-bar-bg {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 9999px;
            height: 12px;
            width: 100%;
            overflow: hidden;
            margin-top: 0.5rem;
        }

        .progress-bar-fill {
            height: 100%;
            border-radius: 9999px;
            transition: width 0.5s ease;
        }

        .action-card {
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 0.5rem;
            padding: 1.25rem;
        }

        .action-card h4 {
            color: var(--accent-blue);
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
        }

        .action-card p {
            font-size: 0.95rem;
            line-height: 1.5;
        }

        .placeholder {
            text-align: center;
            color: var(--text-muted);
            padding: 4rem 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Customer Churn Prediction System</h1>
            <p>Enterprise Machine Learning Retention Intelligence Dashboard</p>
        </header>

        <div class="grid">
            <!-- Customer Input Form -->
            <div class="card">
                <h2>Customer Profile Input</h2>
                <form id="churnForm">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="Age">Age</label>
                            <input type="number" id="Age" value="45" min="18" max="90" required>
                        </div>
                        <div class="form-group">
                            <label for="Gender">Gender</label>
                            <select id="Gender">
                                <option value="Female">Female</option>
                                <option value="Male">Male</option>
                            </select>
                        </div>

                        <div class="form-group">
                            <label for="Tenure">Tenure (Months)</label>
                            <input type="number" id="Tenure" value="6" min="1" max="72" required>
                        </div>
                        <div class="form-group">
                            <label for="MonthlyCharges">Monthly Charges ($)</label>
                            <input type="number" id="MonthlyCharges" value="85.50" step="0.5" required>
                        </div>

                        <div class="form-group">
                            <label for="Contract">Contract Type</label>
                            <select id="Contract">
                                <option value="Month-to-month">Month-to-month</option>
                                <option value="One year">One year</option>
                                <option value="Two year">Two year</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="InternetService">Internet Service</label>
                            <select id="InternetService">
                                <option value="Fiber optic">Fiber optic</option>
                                <option value="DSL">DSL</option>
                                <option value="No">No Internet</option>
                            </select>
                        </div>

                        <div class="form-group">
                            <label for="TechSupport">Tech Support</label>
                            <select id="TechSupport">
                                <option value="No">No</option>
                                <option value="Yes">Yes</option>
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="Complaints">Recent Complaint?</label>
                            <select id="Complaints">
                                <option value="1">Yes (1)</option>
                                <option value="0">No (0)</option>
                            </select>
                        </div>

                        <div class="form-group">
                            <label for="SatisfactionScore">Satisfaction (1-5)</label>
                            <input type="number" id="SatisfactionScore" value="2" min="1" max="5" required>
                        </div>
                        <div class="form-group">
                            <label for="SupportTickets">Support Tickets</label>
                            <input type="number" id="SupportTickets" value="3" min="0" max="10" required>
                        </div>

                        <div class="form-group full">
                            <label for="UsageHours">Monthly Usage Hours</label>
                            <input type="number" id="UsageHours" value="40" min="0" max="400" required>
                        </div>
                    </div>

                    <button type="submit">Predict Churn Risk</button>
                </form>
            </div>

            <!-- Prediction Results -->
            <div class="card">
                <h2>ML Intelligence Output</h2>
                <div id="resultsContent">
                    <div class="placeholder">
                        <p>Fill out the profile on the left and click <strong>Predict Churn Risk</strong> to run real-time ML inference.</p>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('churnForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const payload = {
                Age: parseInt(document.getElementById('Age').value),
                Gender: document.getElementById('Gender').value,
                Tenure: parseInt(document.getElementById('Tenure').value),
                MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
                Contract: document.getElementById('Contract').value,
                InternetService: document.getElementById('InternetService').value,
                TechSupport: document.getElementById('TechSupport').value,
                Complaints: parseInt(document.getElementById('Complaints').value),
                SatisfactionScore: parseInt(document.getElementById('SatisfactionScore').value),
                SupportTickets: parseInt(document.getElementById('SupportTickets').value),
                UsageHours: parseFloat(document.getElementById('UsageHours').value)
            };

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    renderResults(data);
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (err) {
                alert('Failed to connect to server.');
            }
        });

        function renderResults(data) {
            const isChurn = data.Prediction === 'WILL CHURN';
            const badgeClass = isChurn ? 'badge-churn' : 'badge-retain';
            const barColor = isChurn ? '#ef4444' : '#22c55e';
            const probPct = (data['Churn Probability'] * 100).toFixed(1);

            const html = `
                <div class="result-box">
                    <div style="text-align: center;">
                        <span class="status-badge ${badgeClass}">${data.Prediction}</span>
                    </div>

                    <div class="metric-row">
                        <span class="metric-label">Churn Risk Level</span>
                        <span class="metric-val" style="color: ${barColor}">${data['Risk Level']}</span>
                    </div>

                    <div>
                        <div class="metric-row">
                            <span class="metric-label">Estimated Probability</span>
                            <span class="metric-val">${probPct}%</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div class="progress-bar-fill" style="width: ${probPct}%; background: ${barColor}"></div>
                        </div>
                    </div>

                    <div class="metric-row">
                        <span class="metric-label">Confidence Rating</span>
                        <span class="metric-val">${data['Confidence Level']}</span>
                    </div>

                    <div class="action-card">
                        <h4>RECOMMENDED RETENTION STRATEGY</h4>
                        <p>${data['Recommended Action']}</p>
                    </div>
                </div>
            `;

            document.getElementById('resultsContent').innerHTML = html;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        result = predict_customer_churn(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n[INFO] Starting Customer Churn Prediction Server on http://localhost:{port}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
