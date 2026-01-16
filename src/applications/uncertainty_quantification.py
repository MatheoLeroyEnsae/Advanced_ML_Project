"""
"""
import logging
from src.domain.uncertainty import (
    naive_entropy, supervised_approach, build_embeddings, supervised_approach_grid_CV
)
from collections import defaultdict


def job_uncertainty(train_generations, validation_generations, results_dict, config):

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    train_embeddings, train_is_false = build_embeddings(
        train_generations, 'train', results_dict
    )

    validation_embeddings, validation_is_false = build_embeddings(
        validation_generations, 'validation', results_dict
    )

    try:
        logging.info('Training classifier on train embeddings.')
        probabilities = supervised_approach_grid_CV(
            train_embeddings=train_embeddings, is_false=train_is_false, config=config,
            eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
        results_dict['uncertainty_measures']['p_supervised'] = probabilities
        logging.info('classifier is trained')
    except Exception as e:
        logging.error("Cannot train classifier for the reason :")
        print(str(e))

    logging.info("Perplexity calculation")
    entropies = defaultdict(list)

    for idx, tid in enumerate(validation_generations):
        example = validation_generations[tid]
        most_likely_answer = example['most_likely_answer']
        log_likelihood = most_likely_answer['token_log_likelihoods'] 
        entropies['regular_entropy'].append(naive_entropy(log_likelihood))

    return results_dict, entropies