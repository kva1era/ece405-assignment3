from tests.adapters import run_parse_mmlu_response

ex = {
    "subject": "test",
    "question": "q",
    "options": ["a", "b", "c", "d"],
    "answer": "B"
}

print(run_parse_mmlu_response(ex, "The correct answer is B."))
print(run_parse_mmlu_response(ex, "The correct answer is c"))
print(run_parse_mmlu_response(ex, "I do not know"))