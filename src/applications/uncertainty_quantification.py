"""
"""
import logging
from src.domain.uncertainty import naive_entropy, supervised_approach
from src.domain.prompt import is_answerable
from collections import defaultdict
import numpy as np


def job_uncertainty(train_generations, validation_generations, results_dict, config):

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    validation_embeddings, validation_is_true, validation_answerable = [], [], []
    for tid in validation_generations:
        
        most_likely_answer = validation_generations[tid]['most_likely_answer']
        validation_embeddings.append(most_likely_answer['embedding'])
        validation_is_true.append(most_likely_answer['accuracy'])
        validation_answerable.append(is_answerable(validation_generations[tid]))

    validation_is_false = [1.0 - is_t for is_t in validation_is_true]
    results_dict['validation_is_false'] = validation_is_false

    validation_unanswerable = [1.0 - is_a for is_a in validation_answerable]
    results_dict['validation_unanswerable'] = validation_unanswerable
    logging.info('Unanswerable prop on validation: %f', np.mean(validation_unanswerable))

    # Assemble training data for embedding classification.
    train_is_true, train_embeddings, train_answerable = [], [], []
    for tid in train_generations:
        most_likely_answer = train_generations[tid]['most_likely_answer']
        train_embeddings.append(most_likely_answer['embedding'])
        train_is_true.append(most_likely_answer['accuracy'])
        train_answerable.append(is_answerable(train_generations[tid]))
    train_is_false = [0.0 if is_t else 1.0 for is_t in train_is_true]
    train_unanswerable = [0.0 if is_t else 1.0 for is_t in train_answerable]
    logging.info('Unanswerable prop on p_ik training: %f', np.mean(train_unanswerable))

    try:
        logging.info('Starting training p_ik on train embeddings.')
        # Train classifier of correct/incorrect from embeddings.
        logging.info(len(validation_is_false))
        logging.info("**")
        logging.info(len(train_embeddings))
        logging.info("**")
        logging.info(len(validation_embeddings))
        logging.info("**")
        logging.info(len(train_is_false))
        logging.info(validation_is_false)
        p_ik_predictions = supervised_approach(
            train_embeddings=train_embeddings, is_false=train_is_false,
            eval_embeddings=validation_embeddings, eval_is_false=validation_is_false)
        results_dict['uncertainty_measures']['p_ik'] = p_ik_predictions
        logging.info('Finished training p_ik on train embeddings.')
    except Exception as e:
        logging.error("Cannot comput_p_ik_answerable")
        print(str(e))
    
    entropies = defaultdict(list)

    for idx, tid in enumerate(validation_generations):
        example = validation_generations[tid]
        question = example['question']
        context = example['context']
        full_responses = example["responses"]
        most_likely_answer = example['most_likely_answer']

        if not config.use_all_generations:
            log_liks = [r[1] for r in full_responses[:config.use_num_generations]]
        else:
            log_liks = [r[1] for r in full_responses]

    # Length normalization of generation probabilities.
    log_liks_agg = [np.mean(log_lik) for log_lik in log_liks]

    # Compute naive entropy.
    entropies['regular_entropy'].append(naive_entropy(log_liks_agg))
    #results_dict['uncertainty_measures'].update(entropies)

    return results_dict, entropies