"""Prompt format of xFinder (IAAR-Shanghai/xFinder-qwen1505), reproduced verbatim.

Source: https://github.com/IAAR-Shanghai/xFinder (Yu et al., 2025), licensed
CC BY-NC-ND 4.0. Not covered by this repository's license; see THIRD_PARTY_NOTICES.md.
"""

SYSTEM_PROMPT = "You are a help assistant tasked with extracting the precise key answer from given output sentences."
PROMPT_TEMPLATE = "<|System|>:{system_prompt}\n<|User|>:{input_prompt}\n<|Bot|>:"
NO_VALID_ANSWER = "[No valid answer]"
MATH_ANSWER_RANGE = "a(n) number / set / vector / matrix / interval / expression / function / equation / inequality"
STOP_TOKENS = ("<|endoftext|>", "<|end|>", "<|User|>", "<|System|>")
