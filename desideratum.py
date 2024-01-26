# %%
# use abstract base class 
from abc import ABC, abstractmethod
import random 
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
    
class Desideratum(ABC):

    def __init__(self, *args, **kwargs):
        """Creates dataset, stores useful values"""
        self.target_data = None

    @abstractmethod
    def logits_to_target_data(self, logits):
        """Logits -> useful info, e.g. from logits to logit diff"""
        pass
        
    def set_target_data(self, logits):
        """Sets the target data for the desideratum"""
        self.target_data = self.logits_to_target_data(logits)

    @abstractmethod
    def __call__(self, logits):
        """ Returns performance on the desideratum """
        pass 

class TaskSwitchDesideratum(Desideratum):
    """
    Swapping tasks (i.e., changing `+` to `-`) should not effect variable copying.
    """
    
    def __init__(
        self, 
        tokenizer,
        operations, 
        num_samples, 
        var_values=range(10, 100), 
        balance_answers=True,
        device="cuda"
    ):
        super().__init__()
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

        self.to_samples = [sample[0] for sample in samples]
        self.from_samples = [sample[1] for sample in samples]
        self.to_answers = torch.tensor([answer[0] for answer in answers]).to(device)
        self.from_answers = torch.tensor([answer[1] for answer in answers]).to(device)
    
        self.to_answers_logit_idxs = torch.tensor([tokenizer.encode(str(int(target)), add_special_tokens=False)[1] for target in self.to_answers]).to(device)
        self.from_answers_logit_idxs = torch.tensor([tokenizer.encode(str(int(target)), add_special_tokens=False)[1] for target in self.from_answers]).to(device)

        self.tokenized_to_samples = tokenizer(
                self.to_samples, return_tensors="pt",
            ).to(device)
        self.tokenized_from_samples = tokenizer(
                self.from_samples, return_tensors="pt",
            ).to(device) 

    def logits_to_target_data(self, logits):
        final_token_logits = logits[:, -1] # batch x vocab
        to_answer_logits = final_token_logits.gather(1, self.to_answers_logit_idxs.unsqueeze(1)).squeeze(1)
        target_logits = final_token_logits.gather(1, self.from_answers_logit_idxs.unsqueeze(1)).squeeze(1)
        return target_logits - to_answer_logits # broadcast
    
    def set_target_data(self, logits):
        self.target_data = self.logits_to_target_data(logits)

    def __call__(self, logits): # TODO verify
        if self.target_data is None:
            raise ValueError("Target data not set. Please call `set_target_data` with initial logits.")
        
        new_target_data = self.logits_to_target_data(logits)
        return ((new_target_data - self.target_data) ** 2).mean()
        
class ValueSwitchDesideratum(Desideratum):
    """
    Swapping x variable values should change the answer.
    """
    
    def __init__(
        self,
        tokenizer,
        num_samples = 100,
        operations = ["+", "-"],
        var_values = range(10, 100),
        device = "cuda",
    ):
        # NOTE: from_answers are balanced, to_answers are not
        super().__init__()

        ops = [Operation(op) for op in operations]
        targets = []
        for op in ops:
            for x in var_values:
                for y in var_values:
                    targets.append((x, y, op(x, y), op))
       
        # remove negatives
        targets = [target for target in targets if target[2] >= 0]

        to_samples, from_samples = [], []
        to_answers, from_answers = [], []
        for digit in range(1, 10):
            options = [target for target in targets if str(target[2])[0] == str(digit)]
            selected = random.sample(options, num_samples//9)
            for target in selected:
                from_samples.append(f"x = {target[0]}\ny = {target[1]}\nx {target[3].op_str} y = ")
                new_x = random.sample(var_values, 1)[0]
                while new_x == target[0] or target[3](new_x, target[1]) < 0:
                    new_x = random.sample(var_values, 1)[0]
                to_samples.append(f"x = {new_x}\ny = {target[1]}\nx {target[3].op_str} y = ")
                from_answers.append(target[2])
                to_answers.append(target[3](new_x, target[1]))

        self.to_samples = to_samples
        self.from_samples = from_samples
        self.to_answers = torch.tensor(to_answers).to(device)
        self.from_answers = torch.tensor(from_answers).to(device)
                
        self.to_answers_logit_idxs = torch.tensor([tokenizer.encode(str(target), add_special_tokens=False)[1] for target in to_answers]).to(device)
        self.from_answers_logit_idxs = torch.tensor([tokenizer.encode(str(target), add_special_tokens=False)[1] for target in from_answers]).to(device)

        self.tokenized_to_samples = tokenizer(
                to_samples, return_tensors="pt",
            ).to(device)
        self.tokenized_from_samples = tokenizer(
                from_samples, return_tensors="pt",
            ).to(device)
        
    def logits_to_target_data(self, logits):
        final_token_logits = logits[:, -1] # batch x vocab
        to_answer_logits = final_token_logits.gather(1, self.to_answers_logit_idxs.unsqueeze(1)).squeeze(1)
        target_logits = final_token_logits.gather(1, self.from_answers_logit_idxs.unsqueeze(1)).squeeze(1)
        return target_logits - to_answer_logits # broadcast

    def __call__(self, logits):
        target_data = self.logits_to_target_data(logits)
        return (self.target_data - target_data).mean()
    

class CombinedDesideratum(Desideratum):
    """
    Swapping x variable values should change the answer.
    """
    
    def __init__(
        self,
        tokenizer,
        num_samples = 100,
        operations = ["+", "-"],
        var_values = range(10, 100),
        device = "cuda",
    ):
        # NOTE: from_answers are balanced, to_answers are not
        super().__init__()

        ops = [Operation(op) for op in operations]
        targets = []
        for op in ops:
            for x in var_values:
                for y in var_values:
                    targets.append((x, y, op(x, y), op))
       
        # remove negatives
        targets = [target for target in targets if target[2] >= 0]

        to_samples, from_samples = [], []
        to_answers, from_answers = [], []
        for digit in range(1, 10):
            options = [target for target in targets if str(target[2])[0] == str(digit)]
            selected = random.sample(options, num_samples//9)
            for target in selected:
                from_samples.append(f"x = {target[0]}\ny = {target[1]}\nx {target[3].op_str} y = ")
                new_x = random.sample(var_values, 1)[0]
                while new_x == target[0] or target[3](new_x, target[1]) < 0:
                    new_x = random.sample(var_values, 1)[0]
                to_samples.append(f"x = {new_x}\ny = {target[1]}\nx {target[3].op_str} y = ")
                from_answers.append(target[2])
                to_answers.append(target[3](new_x, target[1]))

        self.to_samples = to_samples
        self.from_samples = from_samples
        self.to_answers = torch.tensor(to_answers).to(device)
        self.from_answers = torch.tensor(from_answers).to(device)
                
        self.to_answers_logit_idxs = torch.tensor([tokenizer.encode(str(target), add_special_tokens=False)[1] for target in to_answers]).to(device)
        self.from_answers_logit_idxs = torch.tensor([tokenizer.encode(str(target), add_special_tokens=False)[1] for target in from_answers]).to(device)

        self.tokenized_to_samples = tokenizer(
                to_samples, return_tensors="pt",
            ).to(device)
        self.tokenized_from_samples = tokenizer(
                from_samples, return_tensors="pt",
            ).to(device)
        
    def logits_to_target_data(self, logits):
        final_token_logits = logits[:, -1] # batch x vocab
        to_answer_logits = final_token_logits.gather(1, self.to_answers_logit_idxs.unsqueeze(1)).squeeze(1)
        target_logits = final_token_logits.gather(1, self.from_answers_logit_idxs.unsqueeze(1)).squeeze(1)
        return target_logits - to_answer_logits # broadcast

    def __call__(self, logits):
        target_data = self.logits_to_target_data(logits)
        return (self.target_data - target_data).mean()
