"""
Optional LLM explanation of an analyzer result (not wired into main.py yet).

The LLM only explains; it must not be the source of treatment decisions.
"""
from dotenv import load_dotenv
load_dotenv()

import os

_client = None


def _get_client():
    # Created lazily so a missing GROQ_API_KEY doesn't crash the engine at import time
    global _client
    if _client is None:
        from groq import Groq
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def fast_breathing_note(age_months, bpm):
    """WHO/IMCI fast-breathing cut-offs. Returns '' when not fast or not measured."""
    if bpm is None or age_months is None:
        return ""
    if age_months < 2:
        cutoff = 60
    elif age_months < 12:
        cutoff = 50
    elif age_months < 60:
        cutoff = 40
    else:
        cutoff = 30   # not an IMCI band; general paediatric guide for 5+ years
    if bpm >= cutoff:
        return f"FAST BREATHING (≥{cutoff}/min for age) – IMCI pneumonia sign"
    return ""


def get_medical_explanation(result, age_months, symptoms, respiratory_rate_bpm=None):
    """
    result: the dict returned by analyzer.analyze_audio
    age_months: patient age in months (IMCI bands are defined in months)
    respiratory_rate_bpm: counted by the health worker, if available
    """
    bio = result.get("biomarkers", {})
    diagnosis = bio.get("ai_diagnosis", "Unknown")
    prob_p = bio.get("prob_pneumonia", 0.0)
    prob_a = bio.get("prob_asthma", 0.0)
    prob_n = bio.get("prob_normal", 0.0)
    flags = bio.get("acoustic_flags", {})
    danger = bio.get("danger_signs", [])

    rr_text = (f"{respiratory_rate_bpm}/min. {fast_breathing_note(age_months, respiratory_rate_bpm) or 'Not fast for age.'}"
               if respiratory_rate_bpm is not None else "Not counted")

    user_prompt = f"""
Patient:
- Age: {age_months} months
- Reported symptoms: {symptoms or "None provided"}
- IMCI danger signs reported: {", ".join(danger) or "None"}
- Respiratory rate: {rr_text}

AI lung sound analysis (screening model, trained mostly on ADULT recordings):
- Pattern: {diagnosis}  (risk: {result.get("risk_level_output", "Unknown")})
- Pneumonia-like: {int(prob_p * 100)}%, Asthma/wheeze-like: {int(prob_a * 100)}%, Normal: {int(prob_n * 100)}%
- Heuristic sound flags (segments): {flags or "none"}

In under 80 words, explain to a community health worker what the sound analysis suggests and
how confident it is, and remind them which IMCI assessment steps to complete (count breathing,
look for chest indrawing and danger signs). Do not prescribe medication or doses; refer to the
local IMCI protocol and a clinician for treatment.
"""

    system_prompt = (
        "You support community health workers using WHO/IMCI guidelines. You explain AI screening "
        "results cautiously, never overstate certainty, and never replace clinical assessment."
    )

    try:
        chat_completion = _get_client().chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.25,
            max_tokens=180,
        )
        explanation = chat_completion.choices[0].message.content.strip()
        return (
            explanation +
            "\n\n**Important:** This is AI-generated support based on sound analysis only. "
            "Full clinical assessment (including breathing count, oximetry, exam) is essential. "
            "Follow local IMCI protocols and consult a doctor."
        )

    except Exception as e:
        print(f"Groq LLM error: {e}")
        return (
            "AI interpretation unavailable at this time. "
            "Please follow standard IMCI/WHO guidelines: assess for danger signs, "
            "count the breathing rate and look for chest indrawing. Seek medical help urgently if severe."
        )
