def validate_synonym(word: str, syn_candidate: str) -> bool:
    if '_' not in syn_candidate and word.lower() != syn_candidate.lower():
        return True