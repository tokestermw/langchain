"""Select examples based on length."""
import re
from typing import Callable, Dict, List
from pydantic import BaseModel, validator
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.prompt import PromptTemplate

def _get_length_based(text: str) -> int:
    return len(re.split('\n| ', text))

class LengthBasedExampleSelector(BaseExampleSelector, BaseModel):
    """Select examples based on length."""
    examples: List[dict]
    'A list of the examples that the prompt template expects.'
    example_prompt: PromptTemplate
    'Prompt template used to format the examples.'
    get_text_length: Callable[[str], int] = _get_length_based
    'Function to measure prompt length. Defaults to word count.'
    max_length: int = 2048
    'Max length for the prompt, beyond which examples are cut.'
    example_text_lengths: List[int] = []

    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to list."""
        self.examples.append(example)
        string_example = self.example_prompt.format(**example)
        self.example_text_lengths.append(self.get_text_length(string_example))

    @validator('example_text_lengths', always=True)
    def calculate_example_text_lengths(cls, v: List[int], values: Dict) -> List[int]:
        """Calculate text lengths if they don't exist."""
        if v:
            return v
        example_prompt = values['example_prompt']
        get_text_length = values['get_text_length']
        string_examples = [example_prompt.format(**eg) for eg in values['examples']]
        return [get_text_length(eg) for eg in string_examples]

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the input lengths."""
        inputs = ' '.join(input_variables.values())
        remaining_length = self.max_length - self.get_text_length(inputs)
        examples = []
        for (i, example_text_length) in enumerate(self.example_text_lengths):
            if remaining_length <= 0:
                break
            remaining_length -= example_text_length
            if remaining_length >= 0:
                examples.append(self.examples[i])
        return examples