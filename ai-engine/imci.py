"""WHO IMCI rules used by the engine (classification only; treatment is left to the local protocol)."""

# General danger signs (IMCI): any one means severe illness -> urgent referral.
GENERAL_DANGER_SIGNS = {"unable_to_drink_or_breastfeed", "vomits_everything", "convulsions",
                        "lethargic_or_unconscious", "stridor", "cyanosis"}
PNEUMONIA_SIGNS = {"chest_indrawing"}


def fast_breathing_cutoff(age_months):
    """Breaths per minute at or above which breathing is 'fast' for the child's age."""
    if age_months is None:
        return None
    if age_months < 2:
        return 60
    if age_months < 12:
        return 50
    if age_months < 60:
        return 40
    return 30   # not an IMCI band (IMCI covers under-5s); general paediatric guide for 5+ years


def is_fast_breathing(age_months, breaths_per_minute):
    """True / False, or None when the breathing rate or age was not given."""
    cutoff = fast_breathing_cutoff(age_months)
    if cutoff is None or breaths_per_minute is None:
        return None
    return breaths_per_minute >= cutoff
