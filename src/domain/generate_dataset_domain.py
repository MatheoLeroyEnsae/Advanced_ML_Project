import logging
import random
from tqdm import tqdm
import torch
import gc
import numpy as np

from src.domain.prompt import get_reference
from src.domain.uncertainty import calculate_p_true


def generate_dataset_domain(
    train_dataset, 
    validation_dataset, 
    remaining_answerable, 
    model,
    metric,
    make_prompt,
    prompt,
    p_true_few_shot_prompt,
    config
):
    for dataset_split in ['train', 'validation']:

        accuracies, generations, results_dict, p_trues = [], {}, {}, []

        if dataset_split == 'train':
            dataset = train_dataset
            possible_indices = remaining_answerable

        else:
            dataset = validation_dataset
            possible_indices = range(0, len(dataset))

        indices = random.sample(possible_indices, min(config.num_samples, len(dataset)))

        it = 0
        for index in tqdm(indices):
            if (it + 1 % 10) == 0:
                gc.collect()
                torch.cuda.empty_cache()
            it += 1

            example = dataset[index]
            question, context = example["question"], example['context']
            generations[example['id']] = {'question': question, 'context': context}
            correct_answer = example['answers']['text']

            current_input = make_prompt(
                context, question, None, config.instruction, True)
            local_prompt = prompt + current_input

            logging.info('Current input: '.ljust(15) + current_input)

            answers_high_T = []

            if dataset_split == 'train' and config.get_training_set_generations_most_likely_only:
                num_generations = 1
            else:
                num_generations = config.n_prediction + 1 
            
            for i in range(num_generations):

                temperature = 0.1 if i == 0 else config.temperature

                predicted_answer, token_log_likelihoods, embedding = model.predict(
                    local_prompt, temperature)
                embedding = embedding.cpu() if embedding is not None else None

                if correct_answer and (i == 0):
                    acc = metric(predicted_answer, example)
                else:
                    acc = 0.0

                if i == 0:
                    logging.info('Iteration ' + str(it) + ':  ' + 80*'#')
                    logging.info('question: '.ljust(15) + question)
                    logging.info('low-t prediction: '.ljust(15) + predicted_answer)
                    logging.info('correct answer: '.ljust(15) + str(correct_answer))
                    logging.info('accuracy: '.ljust(15) + str(acc))

                    accuracies.append(acc)

                    most_likely_answer_dict = {
                        'response': predicted_answer,
                        'token_log_likelihoods': token_log_likelihoods,
                        'embedding': embedding,
                        'accuracy': acc}
                    generations[example['id']].update({
                        'most_likely_answer': most_likely_answer_dict,
                        'reference': get_reference(example)})
                    
                else:
                    logging.info('high-t prediction '.ljust(15) + str(i) + ' : ' + predicted_answer)
                    answers_high_T.append(
                        (predicted_answer, token_log_likelihoods, embedding, acc))

            generations[example['id']]['responses'] = answers_high_T

            if config.compute_p_true and dataset_split == 'validation':
                p_true = calculate_p_true(
                    model, question, most_likely_answer_dict['response'],
                    [r[0] for r in answers_high_T], p_true_few_shot_prompt,
                    hint=config.p_true_hint)
                p_trues.append(p_true)
                logging.info('p_true: %s', p_true)
            
            if dataset_split == "train":
                train_generations = generations
            if dataset_split == "validation":
                validation_generations = generations
                
            accuracy = np.mean(accuracies)
            print(f"Overall {dataset_split} split accuracy: {accuracy}")
            if dataset_split == 'validation':
                results_dict['uncertainty_measures'] = {
                    'p_false':  [1 - p for p in p_trues], 
                    'p_false_fixed':  [1 - np.exp(p) for p in p_trues], 
                }

    return train_generations, validation_generations, results_dict
