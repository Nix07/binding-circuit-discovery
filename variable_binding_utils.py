import random
import matplotlib.pyplot as plt
import numpy as np
import torch

class Operation:
    def __init__(self, op_str):
        self.op_str = op_str 
        if op_str == "-":
            self.op = lambda x, y: x - y
        elif op_str == "+":
            self.op = lambda x, y: x + y
        elif op_str == "*":
            self.op = lambda x, y: x * y
        elif op_str == "/":
            self.op = lambda x, y: x / y
        elif op_str == "%":
            self.op = lambda x, y: x % y
        else:
            raise ValueError(f"Unknown op {self.op}")
    
    def __call__(self, x, y):
        return self.op(x, y)

def get_data_desideratum_1(
    operations = ["+", "-"],
    num_samples = 100,
    var_values = range(10, 100),
    balance_answers = True,
):
    samples = []
    answers = []
    for op_1 in operations:
        o1 = Operation(op_1)
        for op_2 in operations:
            if op_1 == op_2:
                continue
            o2 = Operation(op_2)
            for x in var_values:
                for y in var_values:
                    samples.append(
                        [f"x = {x}\ny = {y}\nx {op_1} y = ", f"x = {x}\ny = {y}\nx {op_2} y = "]
                    )
                    answers.append(
                        [o1(x, y), o2(x, y)]
                    )
    
    # remove negative answers
    samples = [sample for sample, answer in zip(samples, answers) if all([a >= 0 for a in answer])]
    answers = [answer for answer in answers if all([a >= 0 for a in answer])]

    if balance_answers:
        samples_balanced = []
        answers_balanced = []
        for i in range(1, 10):
            options = [idx for idx, right in enumerate(answers) if str(right[0])[0] == str(i)]
            selected = random.sample(options, num_samples//9)
            for idx in selected:
                samples_balanced.append(samples[idx])
                answers_balanced.append(answers[idx])
        samples = samples_balanced
        answers = answers_balanced
    
    else: 
        samples = random.sample(samples, num_samples)
        answers = random.sample(answers, num_samples)

    return samples, answers

            
                

def get_data(
    to_operation = "-",
    from_operations = ["+", "-", "*", "/", "%"],
    min_from_samples = 100,
    var_values = range(10, 100),
    balanced_targets = True,
    change_both_vars = False,
):
    # Get to_sample
    to_op = Operation(to_operation)
    x_to = random.sample(var_values, 1)[0]
    y_to = random.sample([y for y in var_values if y < x_to], 1)[0] # y should be smaller than x
    to_sample = f"x = {x_to}\ny = {y_to}\nx {to_operation} y = "
    to_answer = to_op(x_to, y_to)

    # get from_samples
    from_samples = []
    from_answers = []
    from_targets = [] # target uses the to operation 
    for op in from_operations:
        if change_both_vars:
            samples = [f"x = {x}\ny = {y}\nx {op} y = " for x in var_values for y in var_values]
            answers = [Operation(op)(x, y) for x in var_values for y in var_values]
            targets = [to_op(x, y) for x in var_values for y in var_values]
        else:
            samples = [f"x = {x}\ny = {y_to}\nx {op} y = " for x in var_values]
            answers = [Operation(op)(x, y_to) for x in var_values]
            targets = [to_op(x, y_to) for x in var_values]
        from_samples.extend(samples)
        from_answers.extend(answers)
        from_targets.extend(targets)
    
    # remove samples with negative answers
    from_samples = [sample for sample, answer, target in zip(from_samples, from_answers, from_targets) if answer >= 0 and target >= 0]
    from_answers = [answer for answer, target in zip(from_answers, from_targets) if answer >= 0 and target >= 0]
    from_targets = [target for target, answer in zip(from_targets, from_answers) if answer >= 0 and target >= 0]    

    # remove samples with non-int answers
    from_samples = [sample for sample, answer in zip(from_samples, from_answers) if answer == int(answer)]
    from_answers = [answer for answer in from_answers if answer == int(answer)]
    from_targets = [target for target, answer in zip(from_targets, from_answers) if answer == int(answer)]

    # balance the number of samples per first digit of the answer 
    if balanced_targets:
        samples_per = min_from_samples // 9
        from_prompts_balanced = []
        from_answers_balanced = []
        from_targets_balanced = []
        for i in range(1, 10):
            options = [idx for idx, right in enumerate(from_targets) if str(right)[0] == str(i)]
            selected = random.sample(options, samples_per)
            for idx in selected:
                from_prompts_balanced.append(from_samples[idx])
                from_answers_balanced.append(from_answers[idx])
                from_targets_balanced.append(from_targets[idx])
        from_samples = from_prompts_balanced
        from_answers = from_answers_balanced 
        from_targets = from_targets_balanced
    else:
        from_out = random.sample(from_samples, min_from_samples)
        from_answers = [answer for sample, answer in zip(from_samples, from_answers) if sample in from_out]
        from_targets = [target for sample, target in zip(from_samples, from_targets) if sample in from_out]
        from_samples = from_out

    return to_sample, to_answer, from_samples, from_answers, from_targets

def extract_numeric_substrings(strings):
    numeric_substrings = []
    for s in strings:
        numeric_substring = ""
        for c in s:
            if c.isdigit():
                numeric_substring += c
            else:
                break
        if numeric_substring:
            numeric_substrings.append(int(numeric_substring))
        else:
            numeric_substrings.append(-1)
    return numeric_substrings

def predict_answers(model, tokenizer, prompts, first_digit_only=False, device="cuda") -> list[int]:
    # tokenize prompts
    encoded_prompts = tokenizer.batch_encode_plus(prompts, 
                                                  padding=True, 
                                                  truncation=True, 
                                                  max_length=128, 
                                                  return_tensors="pt")
    encoded_prompts = {key: value.to(device) for key, value in encoded_prompts.items()}
    
    # generate answers
    with torch.no_grad():
        output = model.generate(**encoded_prompts,
                                max_new_tokens=4,
                                do_sample=False,
                                num_beams=1,
                                early_stopping=True)

    # decode answers
    answers = tokenizer.batch_decode(output, skip_special_tokens=True)
    answers = [answer.split(" = ")[3] for answer in answers]
    if first_digit_only:
        # if 31, then we want 3
        answers = [int(str(answer)[0]) for answer in answers]
    else:
        answers = extract_numeric_substrings(answers)
    
    return answers