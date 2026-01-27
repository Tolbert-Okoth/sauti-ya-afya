from dotenv import load_dotenv
load_dotenv() 

import os
from groq import Groq

# Initialize Groq Client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_medical_explanation(biomarkers, age, symptoms, threshold=0.75):
    """
    Phase 4 Upgrade: Multi-Class Reasoning.
    Now interprets Asthma vs. Pneumonia probabilities to guide treatment.
    """
    
    # 1. Unpack DSP Biomarkers (Math)
    bpm = biomarkers.get('respiratory_rate_bpm', 0)
    clean_ratio = biomarkers.get('clean_audio_ratio', 1.0)
    zcr = biomarkers.get('zero_crossing_rate', 0)
    harmonic_ratio = biomarkers.get('harmonic_ratio', 0)
    
    # 2. Unpack Phase 4 AI Biomarkers (Multi-Class Brain)
    ai_diagnosis = biomarkers.get('ai_diagnosis', "Unknown")
    prob_p = biomarkers.get('prob_pneumonia', 0.0)
    prob_a = biomarkers.get('prob_asthma', 0.0)
    prob_n = biomarkers.get('prob_normal', 0.0)
    
    # 3. Contextual Logic
    quality_note = ""
    if clean_ratio < 0.4:
        quality_note = "⚠️ Audio Quality Poor (Excessive Noise)."
    
    rr_context = "Normal"
    if bpm > 50:
        rr_context = "HIGH (Tachypnea) - Danger Sign"

    # 4. Construct the Differential Diagnosis Prompt
    # We now give the LLM the full breakdown to explain "Why"
    user_prompt = f"""
    Patient Profile:
    - Age: {age}
    - Symptoms: {symptoms}
    
    Diagnostic Inputs:
    1. DIGITAL SIGNAL PROCESSING (Math):
       - Respiratory Rate: {bpm} BPM ({rr_context})
       - Wheeze Indicator: {harmonic_ratio:.2f} (High > 0.3 = Musical/Asthma)
       - Crackle Indicator: {zcr:.2f} (High > 0.2 = Fluid/Pneumonia)
       
    2. PHASE 4 AI SPECIALIST (MobileNetV2 Multi-Class):
       - Primary Diagnosis: {ai_diagnosis.upper()}
       - Pneumonia Probability: {int(prob_p * 100)}% (Infection/Fluid)
       - Asthma Probability:    {int(prob_a * 100)}% (Inflammation/Whistle)
       - Normal Probability:    {int(prob_n * 100)}%
    
    Metadata:
    - {quality_note}
    
    Task:
    1. Interpret the "Battle of Probabilities" (Is it clearly Pneumonia, or is Asthma also likely?).
    2. Explain the physical findings (e.g., "The AI detects high-pitched wheezing typical of Asthma").
    3. Recommend Action based on the specific disease:
       - If Asthma: Suggest Nebulization/Bronchodilators.
       - If Pneumonia: Suggest Antibiotics/Referral.
       - If Normal: Suggest Observation.
    
    Constraint: Keep response under 60 words. Be decisive and professional.
    """

    system_prompt = (
        "You are an expert pediatric pulmonologist. "
        "You have a Multi-Class AI tool that distinguishes between Pneumonia (Infection) and Asthma (Obstruction). "
        "Use this distinction to prevent unnecessary antibiotic use."
    )

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile", 
            temperature=0.3,       
        )
        return chat_completion.choices[0].message.content
        
    except Exception as e:
        print(f"LLM Error: {e}")
        return "AI advice unavailable. Follow standard IMCI protocols."