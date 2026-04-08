from src.interfaces.transformers_interface import TransformersModel

class EvaluatorTM(TransformersModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
