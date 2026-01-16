"""
"""

from src.adapter.data import load_dataset_TriviaQA, split_dataset
from src.domain.HugginFaceModel import instantiate_model
from src.domain.prompt import (
    get_make_prompt, in_context_learning, build_prompt_for_multi_generation
)
from src.domain.metric import get_metric
from src.domain.generate_dataset_domain import generate_dataset_domain
from omegaconf import DictConfig
import random
import logging


def job(config: DictConfig, return_bool: bool = True):  # revoir ici 

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    train_dataset, validation_dataset = load_dataset_TriviaQA(dataset_name="trivia_qa", seed=42)
    logging.info("Trivial QA loaded")

    answerable_indices, unanswerable_indices = split_dataset(train_dataset)
    val_answerable, _ = split_dataset(validation_dataset)

    unanswerable_indices = []
    validation_dataset = [validation_dataset[i] for i in val_answerable]

    prompt_indices = random.sample(answerable_indices, config.num_few_shot)
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))

    BRIEF_PROMPTS = {
        'default': "Answer the following question as briefly as possible.\n",
        'chat': 'Answer the following question in a single brief but complete sentence.\n'
    }

    make_prompt = get_make_prompt()
    BRIEF = BRIEF_PROMPTS[config.BRIEF]
    BRIEF_always = config.BRIEF_always
    prompt = in_context_learning(
        train_dataset, prompt_indices, BRIEF, BRIEF_always, make_prompt
    )

    model = instantiate_model(config.model_name, config.model_max_new_tokens)

    metric = get_metric(config.metric)

    p_true_indices = random.sample(answerable_indices, config.p_true_num_fewshot)
    remaining_answerable = list(set(remaining_answerable) - set(p_true_indices))
    p_true_few_shot_prompt, p_true_responses, len_p_true = build_prompt_for_multi_generation(
            model=model, dataset=train_dataset, indices=p_true_indices,
            prompt=prompt, brief=BRIEF,
            brief_always=False,
            make_prompt=make_prompt, num_generations=config.num_generations,
            metric=metric
    )

    train_generations, validation_generations, results_dict = generate_dataset_domain(
        train_dataset,
        validation_dataset,
        remaining_answerable,
        unanswerable_indices,
        model,
        metric,
        make_prompt,
        prompt,
        p_true_few_shot_prompt,
        config
    )

    logging.info("Dataset created")

    return train_generations, validation_generations, results_dict if return_bool else None





