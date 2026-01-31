🩺 SautiYaAfya – AI-Powered Pediatric Respiratory Triage System

<img width="1793" height="827" alt="Screenshot 2026-01-31 233325" src="https://github.com/user-attachments/assets/002c4d8c-c1a7-4ce8-88f1-2aabea187a40" />
<img width="1809" height="878" alt="Screenshot 2026-02-01 002110" src="https://github.com/user-attachments/assets/1c1c3964-b4aa-4f2e-bee5-fa975e8079d6" />
<img width="1823" height="869" alt="Screenshot 2026-02-01 002221" src="https://github.com/user-attachments/assets/4a33cdbd-39a7-4a3d-a152-2b336d6604c1" />
<img width="1821" height="924" alt="Screenshot 2026-02-01 002234" src="https://github.com/user-attachments/assets/2fd8a3e0-e9da-478c-b82d-dd10e9d31de3" />







Developed for the Ministry of Health, Kenya

SautiYaAfya is a defense-grade telemedicine platform designed to assist Community Health Workers (CHWs) and Doctors in diagnosing pediatric respiratory conditions (Pneumonia, Asthma) using AI audio analysis.

It features:

Modern "Ghost Mode" Dark UI

Real-time epidemiology surveillance

Offline capabilities for remote areas

AI-powered audio triage with detailed spectrogram visualization

Modular, scalable AI architecture for future model expansion

📸 Screenshots

Login Portal / Doctor Dashboard

Secure role-based authentication with glassmorphism UI

Live outbreak map, triage queue, and critical statistics

AI Case Review / Audio Analysis

Detailed patient breakdown with AI confidence scores

Real-time spectrogram visualization of lung sounds

✨ Key Features
🛡️ Core System

Role-Based Access Control (RBAC): Admins, Doctors, CHWs have distinct interfaces

Defense-Ready UI: High-contrast "Dark/Ghost Mode" optimized for low-light and battery saving

Bilingual Support: English (UK) & Kiswahili

🤖 AI & Diagnostics

Acoustic Triage: Uses PyTorch, torchaudio, and DSP to analyze lung sounds

Visual Spectrograms: Real-time Mel-Spectrograms for clinical validation

Confidence Scoring: Probability percentages for Pneumonia, Asthma, and Normal patterns

Hybrid AI System: CNN + DSP + LLM combination allows interpretable reasoning

Scalable AI Architecture: Easily add new AI models, augment datasets, or update thresholds without breaking workflows

📊 Surveillance & Analytics

Live Outbreak Map: Interactive map (Leaflet) tracking high-risk clusters across 47 counties

Epidemiology Analytics: Charts visualizing disease burden and regional hotspots

Referral Tracking: End-to-end tracking of high-risk patient referrals

🔌 Connectivity & Sync

Offline Mode: Caches patient data locally (localStorage) when internet is unavailable

Auto-Sync: Automatically uploads records when a secure connection is restored

Device Management: Calibrates microphone sensitivity and noise cancellation levels

🛠️ Tech Stack

Frontend:

React.js (v18), React Router v6

CSS3, Bootstrap 5, Glassmorphism

Data Visualization:

Recharts (Analytics), React-Leaflet (Maps)

Authentication:

Firebase Auth (Email/Pass, Google)

Backend Integration:

FastAPI (AI Engine), Axios (REST API consumption)

Background Tasks for async audio processing

AI Engine:

PyTorch (MobileNetV2 CNN) for classification

torchaudio for DSP features (bandpass filter, Mel-spectrogram, zero-crossing, harmonic ratio)

Groq LLM for human-readable, medical explanations

Modular AI Design:

Swap or retrain models easily

Add new respiratory conditions

Supports multi-class diagnosis & confidence scoring

Future-ready for transfer learning and continual learning pipelines

Misc / Tools:

Python 3.14+, uuid, gc, dotenv

soundfile, PIL, NumPy

🚀 Getting Started
Prerequisites

Node.js (v16 or higher)

npm or yarn

A running backend server (Flask/Django/Node/FastAPI) serving the API

Installation
# Clone the repository
git clone https://github.com/your-username/sauti-ya-afya.git
cd sauti-ya-afya/client

# Install dependencies
npm install

Configure Environment

Create a .env or config.js to set your API URL (default is localhost:5000)

Run the Application
npm start


The app will launch at http://localhost:3000

📖 User Guide
1. System Administration (Admin)

Dashboard: View total authorized personnel

User Management: Add new staff or revoke access

System Config: Adjust AI confidence thresholds and toggle MOH export

2. Clinical Review (Doctor)

Triage Queue: Monitor incoming patient data sorted by risk level

Case Review: Click any patient to open the AI Review Modal, listen to audio, and view the spectrogram

Resolution: Mark cases as resolved or contact the caregiver via WhatsApp integration

3. Field Screening (CHW)

New Screening: Record patient demographics and capture lung sounds

Offline Sync: Save data locally and sync later via the Settings hub

🔮 Future AI Expansion & Roadmap

SautiYaAfya has a modular AI architecture built for scalability and continuous improvement:

Adding New Models / Conditions

Introduce new CNN, transformer, or hybrid audio models

Support additional pediatric respiratory diseases or general lung conditions

Data Augmentation & Transfer Learning

Incorporate larger datasets with balanced sampling

Apply synthetic audio augmentation (noise injection, pitch shift, time-stretch)

Fine-tune models on emerging respiratory sounds

Enhanced DSP & Explainability

Expand signal processing features: zero-crossing rate, harmonic ratios, spectral flux

Improve spectrogram visualizations for clinical validation

Provide interpretable AI explanations via LLM or rule-based reasoning

Integration with External APIs

Connect to national epidemiology databases for real-time updates

Integrate with other telemedicine platforms for patient referral tracking

Continuous Learning & CI/CD

Auto-retrain models on newly collected labeled data

Support multiple deployment environments: edge devices, mobile apps, cloud servers

Goal: Keep SautiYaAfya adaptable, extensible, and ready for contributions from AI researchers, clinicians, and developers.

🤝 Contributing

We welcome contributions to:

Improve diagnostic accuracy

Enhance UI/UX accessibility

Add AI features and new models

Steps:

# Fork the Project
git checkout -b feature/AmazingFeature

# Commit Changes
git commit -m "feat: Add some AmazingFeature"

# Push & Create Pull Request
git push origin feature/AmazingFeature

📜 License

Distributed under the MIT License. See LICENSE for more information.

👨‍💻 Credits & Contact

Lead Developer: Tolbert Okoth

Organization: SautiYaAfya Research / Ministry of Health (Kenya)

Version: v2.0.4 (Defense Build)

Medical Disclaimer:
SautiYaAfya is a clinical decision support tool. Results are preliminary and must be verified by a qualified clinical officer.

This README now clearly communicates:

Project purpose & tech

How to run and contribute

Future AI scalability with roadmap

Medical and technical professionalism
