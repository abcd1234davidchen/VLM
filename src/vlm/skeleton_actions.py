"""Reviewed English labels and prompt contract for NTU RGB+D 60 actions."""

from __future__ import annotations


# A060 is intentionally excluded from the Phase 1 teacher contract.
PHASE1_ACTION_LABELS = (
    "drinking water", "eating a meal or snack", "brushing teeth", "brushing hair",
    "dropping an object", "picking up an object", "throwing an object", "sitting down",
    "standing up from a chair", "clapping", "reading", "writing", "tearing up paper",
    "putting on a jacket", "taking off a jacket", "putting on a shoe", "taking off a shoe",
    "putting on glasses", "taking off glasses", "putting on a hat or cap",
    "taking off a hat or cap", "cheering", "waving a hand", "kicking an object",
    "reaching into a pocket", "hopping on one foot", "jumping up", "making or answering a phone call",
    "playing with a phone or tablet", "typing on a keyboard", "pointing to something with a finger",
    "taking a selfie", "checking the time on a watch", "rubbing both hands", "nodding or bowing",
    "shaking the head", "wiping the face", "saluting", "putting the palms together",
    "crossing the hands in front", "sneezing or coughing", "staggering", "falling down",
    "touching the head from a headache", "touching the chest from chest pain",
    "touching the back from back pain", "touching the neck from neck pain", "feeling nauseous or vomiting",
    "using a fan", "punching or slapping another person", "kicking another person",
    "pushing another person", "patting another person on the back", "pointing at another person",
    "hugging another person", "giving an object to another person", "touching another person's pocket",
    "shaking hands", "walking toward each other",
)

PHASE1_PROMPT_TEMPLATES = (
    "{label}",
    "a video of a person {label}",
    "a person is {label}",
)


def phase1_prompts() -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(template.format(label=label) for template in PHASE1_PROMPT_TEMPLATES)
        for label in PHASE1_ACTION_LABELS
    )
