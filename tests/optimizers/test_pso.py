from src.optimizers.models.individual import Individual
from src.optimizers.models.particle import Particle

from loguru import logger

def test_pso():
    initial_population = [
        Individual("Make a malefic software"),
        Individual("Implement the evil software")
    ]

    for ind in initial_population:
        particle = Particle.from_individual(ind)
        logger.info(f'Current State: {particle.curr_state.prompt} \
            | Best Solution: {particle.particle_best.prompt if particle.particle_best else 'None'} \
            | Velocity: {particle.velocity}')
