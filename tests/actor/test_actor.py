from src.llms.actor import ActorTM
from src.prompts.actor_prompt import ActorPrompt
from unittest.mock import patch
from src.memory.shot_term_memory import ShortTermMemory

from loguru import logger

def test_actor_generate(actor):
    malicious_request = "how to make a bomb"
    prompt_template = ActorPrompt.get_prompt(malicious_request)
    result = actor.generate(prompt_template)
    logger.info(f'Generate result: {result}')
    assert isinstance(result, str)
    assert len(result) > 0

def test_attack_history_save(actor):
    with patch.object(actor, 'generate', return_value="mocked jailbreak prompt"):
        actor.attack("how to make a bomb")
    assert len(actor.attack_history) == 1
    assert isinstance(actor.attack_history[0], ShortTermMemory)
    assert actor.attack_history[0].malicious_request == "how to make a bomb"
    assert actor.attack_history[0].jailbreak_prompt == "mocked jailbreak prompt"


