"""
API run module that provides a unified interface for processing answers.
This module re-exports functions from hf_run.py with the expected names.
"""
from typing import List, Optional
from .hf_run import (
    process_mcq_answers,
    process_written_math_answers,
    process_true_false_answers,
)


async def process_answers(responses: List[str], choices: Optional[List] = None) -> List[int]:
    """
    Process multiple choice question answers.
    
    Args:
        responses: List of response strings from the model
        choices: Optional list of choices (currently not used, but kept for API compatibility)
    
    Returns:
        List of answer indices (0-4 for a-e, 5 for no answer)
    """
    # The choices parameter is kept for API compatibility but not currently used
    # The hf_run.process_mcq_answers function doesn't need choices
    return await process_mcq_answers(responses)


# Re-export other functions directly
__all__ = [
    "process_answers",
    "process_written_math_answers",
    "process_true_false_answers",
]

