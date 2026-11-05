"""Exact generation instructions copied from the registered API runner."""
VARIANTS = {'concise': 'Give a concise worked solution. Retain necessary calculations or decisions; remove repetition.', 'explicit': 'Give an explicit worked solution for a small student model. Explain the necessary intermediate steps clearly.'}
FORMAT = 'Write the worked solution as ordinary response text, followed by exactly one final line in the original task format: Final answer: <answer>. Do not emit XML thinking tags. Do not discuss these instructions.'
