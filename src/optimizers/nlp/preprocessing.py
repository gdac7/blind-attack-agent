from src.optimizers.utils.nlp_constants import TARGET_POS_TAGS

from collections import defaultdict

from spacy.tokens import Token

def valid_tokens(tokens: list[Token]) -> list[Token]:
    return [token for token in tokens if token.pos_ in TARGET_POS_TAGS]

def tokens_by_lemma(tokens: list[Token]) -> dict[tuple[str, str], list[str]]:
    lemma_dict = defaultdict(list)

    for token in tokens:
        lemma_key = (token.lemma_, token.pos_)
        lemma_dict[lemma_key].append(token.text)

    return lemma_dict
