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
        instruction: str,
        include_instruction: bool
    ) -> str:
        parts = [instruction] if include_instruction else []

        if context:
            parts.append(f"Context: {context}")

        parts.append(f"Question: {question}")

        parts.append(f"Answer: {answer}" if answer else "Answer:")

        return "\n".join(parts) + ("\n\n" if answer else "")

    return make_prompt


def in_context_learning(
    dataset: Dataset,
    prompt_indices: list[int],
    instruction: str,
    include_instruction: bool,
    make_prompt: Callable[[Optional[str], str, Optional[str], str, bool], str]
) -> str:
    """Given a dataset and indices, construct a fewshot prompt."""
    prompt_parts = [
        make_prompt(
            context=dataset[i]["context"],
            question=dataset[i]["question"],
            answer=dataset[i]["answers"]["text"][0],
            instruction=instruction,
            include_instruction=include_instruction
        )
        for i in prompt_indices
    ]

    return "".join(prompt_parts)


def get_reference(example):
    if 'answers' not in example:
        example = example['reference']
    answers = example['answers']
    answer_starts = answers.get('answer_start', [])
    return {'answers': {'answer_start': answer_starts, 'text': answers['text']}, 'id': example['id']}


def build_prompt_for_multi_generation(
        *, model, dataset, indices, prompt, instruction, include_instruction, make_prompt,
        n_prediction, metric):
    """Build few  prompt for p_true uncertainty metric."""
    few_prompts_list = []
    all_answers = dict()
    for index, value in enumerate(indices):
        prompt_list = []
        example = dataset[value]
        question = example["question"]
        context = example["context"]
        if index != 0:
            prompt_list += ['\n']
        prompt_list += ['Question: ' + question]
        prompt_list += ['\nBrainstormed Answers: ']
        question_prompt = make_prompt(context, question, None, instruction, include_instruction)
        local_prompt = prompt + question_prompt

        responses = []
        for j in range(n_prediction + 1):

            if j == 0:
                temperature = 0.1
            else:
                temperature = 1.0

            response, _, _ = model.predict(local_prompt, temperature)
            logging.info('P_TRUE > Response: '.ljust(25) + response)

            responses.append(response)
            prompt_list += [f'{response.strip()} \n']
            if j == 0:
                most_likely_response = response
                is_correct = metric(response, example)
                answers = [answer for answer in example['answers']['text']]
                logging.info('P_TRUE > LOW-T > true answer: '.ljust(35) + str(answers))
                logging.info('P_TRUE > LOW-T > accuracy: '.ljust(35) + str(is_correct))

        all_answers[value] = dict(
            responses=responses, most_likely_response=most_likely_response,
            is_correct=is_correct)

        prompt_list += ['Possible answer: ' + most_likely_response + '\n']
        prompt_list += ['Is the possible answer:\n']
        prompt_list += ['A) True\n']
        prompt_list += ['B) False\n']
        prompt_list += ['The possible answer is:']
        prompt_list += [' A' if is_correct else ' B']

        prompt_len = len(model.tokenizer.encode(''.join(few_prompts_list + prompt_list)))
        input_len_max = prompt_len + n_prediction * model.max_new_tokens + 200

        if input_len_max < model.token_limit:
            few_prompts_list.extend(prompt_list)
        else:
            logging.warning('p_true prompt truncated (length limit) at %d.', index)
            break

    return ''.join(few_prompts_list), all_answers, index


def is_answerable(generation):
    return len(generation['reference']['answers']['text']) > 0
