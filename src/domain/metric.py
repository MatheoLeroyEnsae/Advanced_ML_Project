from evaluate import load
from typing import Callable
from src.domain.prompt import get_reference

# revoir 


def get_metric(metric: str) -> Callable[..., float]:
    if metric == 'squad':

        squad_metric = load("squad_v2")

        def metric_fct(response, example, *args, **kwargs):
            if 'id' in example:
                exid = example['id']
            elif 'id' in example['reference']:
                exid = example['reference']['id']
            else:
                raise ValueError

            prediction = {'prediction_text': response, 'no_answer_probability': 0.0, 'id': exid}
            results = squad_metric.compute(
                predictions=[prediction],
                references=[get_reference(example)])
            return 1.0 if (results['f1'] >= 50.0) else 0.0
    return metric_fct
    
