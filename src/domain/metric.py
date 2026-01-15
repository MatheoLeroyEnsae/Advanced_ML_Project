from evaluate import load
from typing import Callable
from src.domain.prompt import get_reference


def get_metric(metric: str) -> Callable[..., float]:
    if metric == 'squad':

        squad_metric = load("squad_v2")

        def metric_fct(response, instance):
            _id = instance['id'] if 'id' in instance else instance['reference']['id']

            dic_prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': _id}
            results = squad_metric.compute(
                predictions=[dic_prediction],
                references=[get_reference(instance)])
            return 1.0 if (results['f1'] >= 50.0) else 0.0
    return metric_fct
    
