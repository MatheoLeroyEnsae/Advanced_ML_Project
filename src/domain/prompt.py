"""_summary_
"""

from typing import Callable, Optional


def get_make_prompt() -> Callable[[Optional[str], str, Optional[str], str, bool], str]:
    def make_prompt(
        context: Optional[str],
        question: str, answer: Optional[str],
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
