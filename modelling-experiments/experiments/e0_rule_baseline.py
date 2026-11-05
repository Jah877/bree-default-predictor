"""E0: authoritative decisions and scores saved by Bree's generator.

'Flagged for review' is NOT a denial. The primary comparison counts either
review or denial as *intervention*, while a secondary stricter comparison
counts only outright denials. The model produces a binary intervention policy.
"""
PRIMARY_POLICY = "intervention"
ACTION_SETS = {
    "E0_intervention": {"denied", "flagged_for_review"},
    "E0_denial_only": {"denied"},
}
