"""
Example plugin for LocalAI Copilot.

Every plugin must expose a `run_skill(input_text: str) -> str` function.
Plugins are sandboxed — they should not make network calls or access
files outside the project directory.
"""


SKILL_NAME = "Word Counter"
SKILL_DESCRIPTION = "Counts words and characters in the provided text."


def run_skill(input_text: str) -> str:
    """
    Count words and characters in input_text.

    Args:
        input_text: The text to analyse.

    Returns:
        A summary string with word and character counts.
    """
    words = input_text.split()
    word_count = len(words)
    char_count = len(input_text)
    char_no_spaces = len(input_text.replace(" ", ""))
    return (
        f"Word count:      {word_count}\n"
        f"Characters:      {char_count}\n"
        f"Chars (no space): {char_no_spaces}"
    )
