# Customer Complaint Prioritization System 🚨

An end-to-end NLP system that automatically analyzes and prioritizes large volumes of customer complaints using Transformer-based models, enabling support teams to identify urgent issues faster and reduce response time.

This project focuses on **complaint triaging at scale**, not just sentiment analysis.

---

## 📌 Why This Project Exists

In real-world customer support systems, teams receive **thousands of complaints daily** from:
- emails
- chat systems
- feedback forms
- social media

Manually reading and prioritizing these complaints:
- is slow
- is inconsistent
- leads to delayed responses for critical issues

This system automates that process.

---

## 🎯 What This System Does (Clearly)

Given a **bulk list of customer complaints**, the system:

1. Analyzes each complaint using a **fine-tuned Transformer (DistilBERT)**
2. Detects the emotional tone of the message
3. Applies **priority rules** on top of model output
4. Automatically separates complaints into:
   - **High Priority** (urgent, angry, high confidence)
   - **Low Priority** (non-urgent or neutral)

The result is a **ready-to-use prioritized dataset** for support teams.

---

## 🧠 Why This Is NOT “Just Sentiment Analysis”

Most sentiment projects stop at:
> “This message is negative”

This project goes further by:
- processing complaints **in bulk**
- combining **ML predictions + business logic**
- producing **actionable outputs**
- simulating real customer support workflows

This reflects how NLP is actually used in industry.

---

## ⚙️ Priority Assignment Logic

A complaint is marked as **High Priority** if:
- sentiment = `negative`
- confidence ≥ threshold (e.g. 0.8)

All others are marked as **Low Priority**.

This logic is:
- transparent
- configurable
- business-driven (not model-only)

---
## 🏗️ System Architecture


Input File (JSON / CSV)
        ↓
FastAPI Backend
        ↓
Transformer NLP Model (DistilBERT)
        ↓
Priority Logic
        ↓
Sorted Output Files (ZIP)
📂 Input Format
JSON Input
[
  {
    "id": 1,
    "message": "My internet has been down for 3 days and no one is responding"
  }
]

CSV Input

Must contain a column named message

📤 Output

The system returns a ZIP file containing:

high_priority_complaints.json

low_priority_complaints.json

summary.json

Example summary
{
  "total_complaints": 30,
  "high_priority": 11,
  "low_priority": 19
}

🖥️ User Interface

Simple upload-based UI

Supports JSON and CSV files

Displays processing summary

Downloads structured results instantly

A demo video showing the UI and workflow is included.

▶️ Demo

📁 Demo input file: demo_complaints.json

🎥 Demo video: screen-recorded and shared on LinkedIn

The demo shows:

Uploading bulk complaints

Automated analysis

Priority-based sorting

Downloaded results

🛠️ Tech Stack
NLP / ML

PyTorch

Hugging Face Transformers

DistilBERT (fine-tuned)

Backend

FastAPI

Frontend

HTML, CSS, JavaScript

Model Hosting

Hugging Face Hub

▶️ Running Locally
pip install -r requirements.txt
uvicorn app.main:app --reload


Access:

UI → http://localhost:8000

API docs → http://localhost:8000/docs

📌 Real-World Use Cases

Customer support triaging

Telecom & banking complaint handling

CRM automation

Feedback prioritization

Social media issue monitoring

🚀 Future Enhancements

Multi-class issue categorization

SLA-based priority scoring

Dashboard analytics

Database integration

Authentication and role-based access

👤 Author

Hrishi Shah
Machine Learning | NLP | Applied AI

This project demonstrates:

real-world NLP application design

Transformer-based modeling

production-oriented backend architecture

business-driven ML logic




