"""_summary_ revoir get reference
"""

from typing import Callable, Optional
from datasets import Dataset
import logging


def get_make_prompt() -> Callable[[Optional[str], str, Optional[str], str, bool], str]:
    def make_prompt(
        context: Optional[str],
        question: str,
        answer: Optional[str],
        brief: str,
        brief_always: bool
    ) -> str:

        prompt = ''
        if brief_always:
            prompt += brief
        if (context is not None):
            prompt += f"Context: {context}\n"
        prompt += f"Question: {question}\n"
        if answer:
            prompt += f"Answer: {answer}\n\n"
        else:
            prompt += 'Answer:'
        return prompt

    return make_prompt


def build_prompt_from_indices(
    dataset: Dataset,
    prompt_indices: list[int],
    brief: str,
    brief_always: bool,
    make_prompt: Callable[[Optional[str], str, Optional[str], str, bool], str]
) -> str:
    """Given a dataset and indices, construct a fewshot prompt."""
    prompt = brief if not brief_always else ''

    for instance_index in prompt_indices:

        instance = dataset[instance_index]
        context = instance["context"]
        question = instance["question"]
        answer = instance["answers"]["text"][0]

        prompt = prompt + make_prompt(context, question, answer, brief, brief_always)

    return prompt


def get_reference(example):
    if 'answers' not in example:
        example = example['reference']
    answers = example['answers']
    answer_starts = answers.get('answer_start', [])
    return {'answers': {'answer_start': answer_starts, 'text': answers['text']}, 'id': example['id']}


def build_prompt_for_multi_generation(
        *, model, dataset, indices, prompt, brief, brief_always, make_prompt,
        num_generations, metric):
    """Construct few shot prompt for p_true uncertainty metric."""

    # Call model n_shots many times.
    few_shot_prompt = []
    all_responses = dict()
    for it, i in enumerate(indices):
        prompt_candidate = []
        example = dataset[i]
        question = example["question"]
        context = example["context"]
        if it != 0:
            prompt_candidate += ['\n']
        prompt_candidate += ['Question: ' + question]
        prompt_candidate += ['\nBrainstormed Answers: ']
        current_question = make_prompt(context, question, None, brief, brief_always)
        local_prompt = prompt + current_question
        logging.info('P_TRUE >> Current Question: '.ljust(25) + current_question)

        responses = []
        for j in range(num_generations + 1):

            if j == 0:
                temperature = 0.1
            else:
                temperature = 1.0

            response, _, _ = model.predict(local_prompt, temperature)
            logging.info('P_TRUE >> Current Response: '.ljust(25) + response)

            responses.append(response)
            prompt_candidate += [f'{response.strip()} \n']
            if j == 0:
                # Save most likely response and compute correctness metric for it.
                most_likely_response = response
                is_correct = metric(response, example, model)
                answers = [answer for answer in example['answers']['text']]
                logging.info('P_TRUE >> LOW-T >> true answer: '.ljust(35) + str(answers))
                logging.info('P_TRUE >> LOW-T >> acc: '.ljust(35) + str(is_correct))

        all_responses[i] = dict(
            responses=responses, most_likely_response=most_likely_response,
            is_correct=is_correct)

        prompt_candidate += ['Possible answer: ' + most_likely_response + '\n']
        prompt_candidate += ['Is the possible answer:\n']
        prompt_candidate += ['A) True\n']
        prompt_candidate += ['B) False\n']
        prompt_candidate += ['The possible answer is:']
        prompt_candidate += [' A' if is_correct else ' B']

        prompt_len = len(model.tokenizer.encode(''.join(few_shot_prompt + prompt_candidate)))
        # At test time, get a maximum of `num_generations * model.token_limit` extra tokens
        # 200 buffer for question and 'Possible Answer'.
        max_input_len = prompt_len + num_generations * model.max_new_tokens + 200

        if max_input_len < model.token_limit:
            few_shot_prompt.extend(prompt_candidate)
        else:
            logging.warning('Cutting of p_true prompt at length %d.', it)
            break

    return ''.join(few_shot_prompt), all_responses, it


def is_answerable(generation):
    return len(generation['reference']['answers']['text']) > 0
