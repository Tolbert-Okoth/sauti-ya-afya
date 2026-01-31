from dotenv import load_dotenv
load_dotenv()

import os
from groq import Groq

# Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_medical_explanation(biomarkers, age, symptoms, threshold=0.75):
    """
    Upgraded: Stronger antibiotic stewardship, age-specific RR context,
    clearer differential between asthma vs pneumonia, IMCI-aligned logic.
    """
    
    # 1. Extract key biomarkers
    ai_diagnosis = biomarkers.get('ai_diagnosis', "Unknown")
    prob_p = biomarkers.get('prob_pneumonia', 0.0)
    prob_a = biomarkers.get('prob_asthma', 0.0)
    prob_n = biomarkers.get('prob_normal', 0.0)
    
    bpm = biomarkers.get('respiratory_rate_bpm', 0)
    clean_ratio = biomarkers.get('clean_audio_ratio', 1.0)
    zcr = biomarkers.get('zero_crossing_rate', 0.0)
    harmonic_ratio = biomarkers.get('harmonic_ratio', 0.0)
    
    # 2. Age-appropriate RR interpretation (WHO/IMCI pediatric thresholds)
    rr_note = "Normal RR"
    tachypnea_note = ""
    if age < 1:          # <12 months
        if bpm >= 50:
            tachypnea_note = "TACHYPNEA (≥50 bpm) – possible pneumonia danger sign"
        elif bpm > 60:
            tachypnea_note = "SEVERE TACHYPNEA – urgent evaluation"
    elif 1 <= age <= 5:  # 12–59 months
        if bpm >= 40:
            tachypnea_note = "TACHYPNEA (≥40 bpm) – possible pneumonia"
    else:
        if bpm > 30:
            tachypnea_note = "Elevated RR – consider lower airway issue"
    
    quality_note = "Good audio quality."
    if clean_ratio < 0.50:
        quality_note = "⚠️ LOW AUDIO QUALITY – interpretation limited."
    elif clean_ratio < 0.70:
        quality_note += " Moderate noise may affect reliability."

    # 3. Construct precise, stewardship-focused prompt
    user_prompt = f"""
You are an expert pediatric pulmonologist following WHO/IMCI guidelines and antibiotic stewardship principles.

Patient:
- Age: {age} years
- Reported symptoms: {symptoms or "None provided"}

AI Lung Sound Analysis (MobileNetV2 multi-class):
- Primary pattern: {ai_diagnosis.upper()}
- Pneumonia probability: {int(prob_p * 100)}%  (suggests crackles/fluid/infection)
- Asthma probability:    {int(prob_a * 100)}%  (suggests wheeze/obstruction)
- Normal probability:    {int(prob_n * 100)}%

DSP Features:
- Respiratory rate: {bpm} bpm  → {tachypnea_note or rr_note}
- Wheeze/harmonic indicator: {harmonic_ratio:.2f}  (high >0.35 → more likely asthma-like)
- Crackle/chaotic indicator: {zcr:.2f}             (high >0.15–0.20 → more likely crackles/pneumonia)

Audio quality: {quality_note}

Task (be decisive, <80 words):
1. Interpret probability balance: Is pneumonia clearly dominant (>65–70% + tachypnea/crackle signs), or is asthma / viral / normal more likely?
2. Explain key acoustic findings briefly.
3. Action recommendation:
   - Asthma dominant → bronchodilators/nebulization (salbutamol), avoid routine antibiotics.
   - Pneumonia dominant (esp. with tachypnea/danger signs) → antibiotics (amoxicillin preferred) + urgent referral if severe.
   - Normal/low confidence → supportive care, observe, re-assess.
   - Always: This is AI support only – clinical judgment + exam required.

Prioritize avoiding unnecessary antibiotics unless bacterial pneumonia strongly indicated.
"""

    system_prompt = (
        "You are a pediatric pulmonologist expert in IMCI/WHO guidelines. "
        "Differentiate asthma (usually viral-triggered wheeze – no routine antibiotics) "
        "from bacterial pneumonia (crackles, tachypnea – needs antibiotics). "
        "Promote rational antibiotic use to reduce resistance."
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",  # or mixtral/large if available
            temperature=0.25,          # lower for more consistent, guideline-adherent output
            max_tokens=180,
        )
        explanation = chat_completion.choices[0].message.content.strip()
        
        # Safety wrapper
        return (
            explanation + 
            "\n\n**Important:** This is AI-generated support based on sound analysis only. "
            "Full clinical assessment (including auscultation, oximetry, exam) is essential. "
            "Follow local IMCI protocols and consult a doctor."
        )
        
    except Exception as e:
        print(f"Groq LLM error: {e}")
        return (
            "AI interpretation unavailable at this time. "
            "Please follow standard IMCI/WHO guidelines: assess for danger signs, "
            "fast breathing, chest indrawing; treat suspected pneumonia with amoxicillin "
            "if indicated, and use salbutamol for wheeze. Seek medical help urgently if severe."
        )