import random
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM    

def generate_response(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text

class EmbaseNLI(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./Inference_Embase_samples.json')
        self.data = [{'instruction': entry['instruction'], 'premise': entry['input']['premise'].encode('utf-8').decode('unicode-escape'), 'hypothesis': entry['input']['hypothesis'], 'output': entry['output']} for entry in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def has_special_characters(self, text):
        return any(ord(char) > 127 for char in text)

    def read_and_filter(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        filtered_list = [item for item in data if not (
            self.has_special_characters(item['input']['premise']) or
            self.has_special_characters(item['input']['hypothesis'])
        )]
        return filtered_list


def passive_inference(model, tokenizer, mode, prompt_format):

    random.seed(0)
    testset = EmbaseNLI().data

    filename = f"./Embase_NLI_passive_{mode}_{prompt_format}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):

            premise = sample['premise']
            hypothesis = sample['hypothesis']
            label = sample['output']

            if mode == "0shot":

                if prompt_format == 'simple':

                    passive_prompts = [
                        f'Given [{premise}] can it be assumed that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{premise}] can it be inferred that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{premise}] can it be implied that [{hypothesis}].\nIs it true, false or undetermined?'
                    ]

                elif prompt_format == 'qa':

                    passive_prompts = [
                        f'Question: Given [{premise}] can it be assumed that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{premise}] can it be inferred that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{premise}] can it be implied that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:'
                    ]      

                else: 
                    raise ValueError("Prompt format not supported")     
            
            elif mode == "1shot":
                
                icl_sample = random.choice(testset)
                icl_premise = icl_sample['premise']
                icl_hypothesis = icl_sample['hypothesis']
                icl_label = icl_sample['output']

                if prompt_format == 'simple':

                    passive_prompts = [
                        f'Given [{icl_premise}] can it be assumed that [{icl_hypothesis}].\nIs it true, false or undetermined?\n\n{icl_label}\n\nGiven [{premise}] can it be assumed that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise}] can it be inferred that [{icl_hypothesis}].\nIs it true, false or undetermined?\n\n{icl_label}\n\nGiven [{premise}] can it be inferred that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise}] can it be implied that [{icl_hypothesis}].\nIs it true, false or undetermined?\n\n{icl_label}\n\nGiven [{premise}] can it be implied that [{hypothesis}].\nIs it true, false or undetermined?'
                    ]

                elif prompt_format == 'qa':

                    passive_prompts = [
                        f'Question: Given [{icl_premise}] can it be assumed that [{icl_hypothesis}].\nIs it true, false or undetermined?\nAnswer: {icl_label}\n\nQuestion: Given [{premise}] can it be assumed that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise}] can it be inferred that [{icl_hypothesis}].\nIs it true, false or undetermined?\nAnswer: {icl_label}\n\nQuestion: Given [{premise}] can it be inferred that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise}] can it be implied that [{icl_hypothesis}].\nIs it true, false or undetermined?\nAnswer: {icl_label}\n\nQuestion: Given [{premise}] can it be implied that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:'
                    ]

                else: 
                    raise ValueError("Prompt format not supported") 

            elif mode == "3shot":

                random.seed(0)
                icl_sample_1 = random.choice(testset)
                icl_premise_1 = icl_sample_1['premise']
                icl_hypothesis_1 = icl_sample_1['hypothesis']
                icl_label_1 = icl_sample_1['output']

                random.seed(1)
                icl_sample_2 = random.choice(testset)
                icl_premise_2 = icl_sample_2['premise']
                icl_hypothesis_2 = icl_sample_2['hypothesis']
                icl_label_2 = icl_sample_2['output']

                random.seed(4)
                icl_sample_3 = random.choice(testset)
                icl_premise_3 = icl_sample_3['premise']
                icl_hypothesis_3 = icl_sample_3['hypothesis']
                icl_label_3 = icl_sample_3['output']

                if prompt_format == 'simple':

                    passive_prompts = [
                        f'Given [{icl_premise_1}] can it be assumed that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\n\n{icl_label_1}\n\nGiven [{icl_premise_2}] can it be assumed that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\n\n{icl_label_2}\n\nGiven [{icl_premise_3}] can it be assumed that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\n\n{icl_label_3}\n\nGiven [{premise}] can it be assumed that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise_1}] can it be inferred that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\n\n{icl_label_1}\n\nGiven [{icl_premise_2}] can it be inferred that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\n\n{icl_label_2}\n\nGiven [{icl_premise_3}] can it be inferred that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\n\n{icl_label_3}\n\nGiven [{premise}] can it be inferred that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise_1}] can it be implied that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\n\n{icl_label_1}\n\nGiven [{icl_premise_2}] can it be implied that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\n\n{icl_label_2}\n\nGiven [{icl_premise_3}] can it be implied that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\n\n{icl_label_3}\n\nGiven [{premise}] can it be implied that [{hypothesis}].\nIs it true, false or undetermined?',
                    ]

                elif prompt_format == 'qa':

                    passive_prompts = [
                        f'Question: Given [{icl_premise_1}] can it be assumed that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\nAnswer: {icl_label_1}\n\nQuestion: Given [{icl_premise_2}] can it be assumed that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\nAnswer: {icl_label_2}\n\nQuestion: Given [{icl_premise_3}] can it be assumed that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\nAnswer: {icl_label_3}\n\nQuestion: Given [{premise}] can it be assumed that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise_1}] can it be inferred that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\nAnswer: {icl_label_1}\n\nQuestion: Given [{icl_premise_2}] can it be inferred that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\nAnswer: {icl_label_2}\n\nQuestion: Given [{icl_premise_3}] can it be inferred that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\nAnswer: {icl_label_3}\n\nQuestion: Given [{premise}] can it be inferred that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise_1}] can it be implied that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\nAnswer: {icl_label_1}\n\nQuestion: Given [{icl_premise_2}] can it be implied that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\nAnswer: {icl_label_2}\n\nQuestion: Given [{icl_premise_3}] can it be implied that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\nAnswer: {icl_label_3}\n\nQuestion: Given [{premise}] can it be implied that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:'
                    ]

                else: 
                    raise ValueError("Prompt format not supported") 

            else:
                raise ValueError("Mode not supported")

            for j, passive_prompt in enumerate(passive_prompts):
                
                sum = f"Prompt {j}\n"
                f.write(sum)

                single_turn_prompt = f"GPT4 Correct User: {passive_prompt}<|end_of_turn|>GPT4 Correct Assistant:"
                text_output = generate_response(model, tokenizer, single_turn_prompt)

                summary = f"\n{text_output}\n\n{label}\n"
                
                f.write(summary)
                f.write("-\n")

            print(f"Sample {i} done")
            f.write("---\n")


def inter_active_inference(model, tokenizer, mode, prompt_format):

    random.seed(0)
    testset = EmbaseNLI().data

    filename = f"./Embase_NLI_inter_active_{mode}_{prompt_format}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):

            premise = sample['premise']
            hypothesis = sample['hypothesis']
            label = sample['output']

            if mode == "0shot":

                if prompt_format == 'simple':

                    inter_active_prompts = [
                        f'Given [{premise}] can you assume that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{premise}] can you infer that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{premise}] can you imply that [{hypothesis}].\nIs it true, false or undetermined?'
                    ]
                
                elif prompt_format == 'qa':

                    inter_active_prompts = [
                        f'Question: Given [{premise}] can you assume that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{premise}] can you infer that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{premise}] can you imply that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:'
                    ]

                else:
                    raise ValueError("Prompt format not supported")

            elif mode == "1shot":
                
                icl_sample = random.choice(testset)
                icl_premise = icl_sample['premise']
                icl_hypothesis = icl_sample['hypothesis']
                icl_label = icl_sample['output']

                if prompt_format == 'simple':

                    inter_active_prompts = [
                        f'Given [{icl_premise}] can you assume that [{icl_hypothesis}].\nIs it true, false or undetermined?\n\n{icl_label}\n\nGiven [{premise}] can you assume that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise}] can you infer that [{icl_hypothesis}].\nIs it true, false or undetermined?\n\n{icl_label}\n\nGiven [{premise}] can you infer that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise}] can you imply that [{icl_hypothesis}].\nIs it true, false or undetermined?\n\n{icl_label}\n\nGiven [{premise}] can you imply that [{hypothesis}].\nIs it true, false or undetermined?'
                    ]

                elif prompt_format == 'qa':

                    inter_active_prompts = [
                        f'Question: Given [{icl_premise}] can you assume that [{icl_hypothesis}].\nIs it true, false or undetermined?\nAnswer: {icl_label}\n\nQuestion: Given [{premise}] can you assume that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise}] can you infer that [{icl_hypothesis}].\nIs it true, false or undetermined?\nAnswer: {icl_label}\n\nQuestion: Given [{premise}] can you infer that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise}] can you imply that [{icl_hypothesis}].\nIs it true, false or undetermined?\nAnswer: {icl_label}\n\nQuestion: Given [{premise}] can you imply that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:'
                    ]

                else:
                    raise ValueError("Prompt format not supported")

            elif mode == "3shot":

                random.seed(0)
                icl_sample_1 = random.choice(testset)
                icl_premise_1 = icl_sample_1['premise']
                icl_hypothesis_1 = icl_sample_1['hypothesis']
                icl_label_1 = icl_sample_1['output']

                random.seed(1)
                icl_sample_2 = random.choice(testset)
                icl_premise_2 = icl_sample_2['premise']
                icl_hypothesis_2 = icl_sample_2['hypothesis']
                icl_label_2 = icl_sample_2['output']

                random.seed(4)
                icl_sample_3 = random.choice(testset)
                icl_premise_3 = icl_sample_3['premise']
                icl_hypothesis_3 = icl_sample_3['hypothesis']
                icl_label_3 = icl_sample_3['output']

                if prompt_format == 'simple':

                    inter_active_prompts = [
                        f'Given [{icl_premise_1}] can you assume that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\n\n{icl_label_1}\n\nGiven [{icl_premise_2}] can you assume that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\n\n{icl_label_2}\n\nGiven [{icl_premise_3}] can you assume that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\n\n{icl_label_3}\n\nGiven [{premise}] can you assume that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise_1}] can you infer that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\n\n{icl_label_1}\n\nGiven [{icl_premise_2}] can you infer that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\n\n{icl_label_2}\n\nGiven [{icl_premise_3}] can you infer that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\n\n{icl_label_3}\n\nGiven [{premise}] can you infer that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise_1}] can you imply that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\n\n{icl_label_1}\n\nGiven [{icl_premise_2}] can you imply that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\n\n{icl_label_2}\n\nGiven [{icl_premise_3}] can you imply that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\n\n{icl_label_3}\n\nGiven [{premise}] can you imply that [{hypothesis}].\nIs it true, false or undetermined?',
                    ]
                
                elif prompt_format == 'qa':

                    inter_active_prompts = [
                        f'Question: Given [{icl_premise_1}] can you assume that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\nAnswer: {icl_label_1}\n\nQuestion: Given [{icl_premise_2}] can you assume that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\nAnswer: {icl_label_2}\n\nQuestion: Given [{icl_premise_3}] can you assume that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\nAnswer: {icl_label_3}\n\nQuestion: Given [{premise}] can you assume that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise_1}] can you infer that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\nAnswer: {icl_label_1}\n\nQuestion: Given [{icl_premise_2}] can you infer that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\nAnswer: {icl_label_2}\n\nQuestion: Given [{icl_premise_3}] can you infer that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\nAnswer: {icl_label_3}\n\nQuestion: Given [{premise}] can you infer that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise_1}] can you imply that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\nAnswer: {icl_label_1}\n\nQuestion: Given [{icl_premise_2}] can you imply that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\nAnswer: {icl_label_2}\n\nQuestion: Given [{icl_premise_3}] can you imply that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\nAnswer: {icl_label_3}\n\nQuestion: Given [{premise}] can you imply that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                    ]

                else:
                    raise ValueError("Prompt format not supported")

            for j, inter_active_prompt in enumerate(inter_active_prompts):
                
                sum = f"Prompt {j}\n"
                f.write(sum)

                single_turn_prompt = f"GPT4 Correct User: {inter_active_prompt}<|end_of_turn|>GPT4 Correct Assistant:"
                text_output = generate_response(model, tokenizer, single_turn_prompt)

                summary = f"\n{text_output}\n\n{label}\n"
                
                f.write(summary)
                f.write("-\n")

            print(f"Sample {i} done")
            f.write("---\n")


def ind_inference(model, tokenizer, mode, prompt_format):

    random.seed(0)
    testset = EmbaseNLI().data

    filename = f"./Embase_NLI_ind_{mode}_{prompt_format}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):

            premise = sample['premise']
            hypothesis = sample['hypothesis']
            label = sample['output']

            if mode == "0shot":

                if prompt_format == 'simple':

                    ind_prompts = [
                        f'Given [{premise}] you can assume that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{premise}] you can infer that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{premise}] you can imply that [{hypothesis}].\nIs it true, false or undetermined?'
                    ]
                
                elif prompt_format == 'qa':

                    ind_prompts = [
                        f'Question: Given [{premise}] you can assume that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{premise}] you can infer that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{premise}] you can imply that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:'
                    ]

                else:
                    raise ValueError("Prompt format not supported")

            elif mode == "1shot":
                
                icl_sample = random.choice(testset)
                icl_premise = icl_sample['premise']
                icl_hypothesis = icl_sample['hypothesis']
                icl_label = icl_sample['output']

                if prompt_format == 'simple':

                    ind_prompts = [
                        f'Given [{icl_premise}] you can assume that [{icl_hypothesis}].\nIs it true, false or undetermined?\n\n{icl_label}\n\nGiven [{premise}] you can assume that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise}] you can infer that [{icl_hypothesis}].\nIs it true, false or undetermined?\n\n{icl_label}\n\nGiven [{premise}] you can infer that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise}] you can imply that [{icl_hypothesis}].\nIs it true, false or undetermined?\n\n{icl_label}\n\nGiven [{premise}] you can imply that [{hypothesis}].\nIs it true, false or undetermined?'
                    ]

                elif prompt_format == 'qa':

                    ind_prompts = [
                        f'Question: Given [{icl_premise}] you can assume that [{icl_hypothesis}].\nIs it true, false or undetermined?\nAnswer: {icl_label}\n\nQuestion: Given [{premise}] you can assume that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise}] you can infer that [{icl_hypothesis}].\nIs it true, false or undetermined?\nAnswer: {icl_label}\n\nQuestion: Given [{premise}] you can infer that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise}] you can imply that [{icl_hypothesis}].\nIs it true, false or undetermined?\nAnswer: {icl_label}\n\nQuestion: Given [{premise}] you can imply that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:'
                    ]

                else:
                    raise ValueError("Prompt format not supported")

            elif mode == "3shot":

                random.seed(0)
                icl_sample_1 = random.choice(testset)
                icl_premise_1 = icl_sample_1['premise']
                icl_hypothesis_1 = icl_sample_1['hypothesis']
                icl_label_1 = icl_sample_1['output']

                random.seed(1)
                icl_sample_2 = random.choice(testset)
                icl_premise_2 = icl_sample_2['premise']
                icl_hypothesis_2 = icl_sample_2['hypothesis']
                icl_label_2 = icl_sample_2['output']

                random.seed(4)
                icl_sample_3 = random.choice(testset)
                icl_premise_3 = icl_sample_3['premise']
                icl_hypothesis_3 = icl_sample_3['hypothesis']
                icl_label_3 = icl_sample_3['output']

                if prompt_format == 'simple':

                    ind_prompts = [
                        f'Given [{icl_premise_1}] you can assume that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\n\n{icl_label_1}\n\nGiven [{icl_premise_2}] you can assume that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\n\n{icl_label_2}\n\nGiven [{icl_premise_3}] you can assume that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\n\n{icl_label_3}\n\nGiven [{premise}] you can assume that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise_1}] you can infer that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\n\n{icl_label_1}\n\nGiven [{icl_premise_2}] you can infer that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\n\n{icl_label_2}\n\nGiven [{icl_premise_3}] you can infer that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\n\n{icl_label_3}\n\nGiven [{premise}] you can infer that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise_1}] you can imply that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\n\n{icl_label_1}\n\nGiven [{icl_premise_2}] you can imply that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\n\n{icl_label_2}\n\nGiven [{icl_premise_3}] you can imply that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\n\n{icl_label_3}\n\nGiven [{premise}] you can imply that [{hypothesis}].\nIs it true, false or undetermined?',
                    ]

                elif prompt_format == 'qa':

                    ind_prompts = [
                        f'Question: Given [{icl_premise_1}] you can assume that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\nAnswer: {icl_label_1}\n\nQuestion: Given [{icl_premise_2}] you can assume that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\nAnswer: {icl_label_2}\n\nQuestion: Given [{icl_premise_3}] you can assume that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\nAnswer: {icl_label_3}\n\nQuestion: Given [{premise}] you can assume that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise_1}] you can infer that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\nAnswer: {icl_label_1}\n\nQuestion: Given [{icl_premise_2}] you can infer that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\nAnswer: {icl_label_2}\n\nQuestion: Given [{icl_premise_3}] you can infer that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\nAnswer: {icl_label_3}\n\nQuestion: Given [{premise}] you can infer that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise_1}] you can imply that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\nAnswer: {icl_label_1}\n\nQuestion: Given [{icl_premise_2}] you can imply that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\nAnswer: {icl_label_2}\n\nQuestion: Given [{icl_premise_3}] you can imply that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\nAnswer: {icl_label_3}\n\nQuestion: Given [{premise}] you can imply that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                    ]

                else:
                    raise ValueError("Prompt format not supported")

            for j, ind_prompt in enumerate(ind_prompts):
                
                sum = f"Prompt {j}\n"
                f.write(sum)

                single_turn_prompt = f"GPT4 Correct User: {ind_prompt}<|end_of_turn|>GPT4 Correct Assistant:"
                text_output = generate_response(model, tokenizer, single_turn_prompt)

                summary = f"\n{text_output}\n\n{label}\n"
                
                f.write(summary)
                f.write("-\n")

            print(f"Sample {i} done")
            f.write("---\n")


def imp_inference(model, tokenizer, mode, prompt_format):

    random.seed(0)
    testset = EmbaseNLI().data

    filename = f"./Embase_NLI_imp_{mode}_{prompt_format}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):

            premise = sample['premise']
            hypothesis = sample['hypothesis']
            label = sample['output']

            if mode == "0shot":

                if prompt_format == 'simple':

                    imp_prompts = [
                        f'Given [{premise}] tell me if you can assume that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{premise}] tell me if you can infer that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{premise}] tell me if you can imply that [{hypothesis}].\nIs it true, false or undetermined?'
                    ]

                elif prompt_format == 'qa':

                    imp_prompts = [
                        f'Question: Given [{premise}] tell me if you can assume that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{premise}] tell me if you can infer that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{premise}] tell me if you can imply that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:'
                    ]

                else:
                    raise ValueError("Prompt format not supported")

            elif mode == "1shot":
                
                icl_sample = random.choice(testset)
                icl_premise = icl_sample['premise']
                icl_hypothesis = icl_sample['hypothesis']
                icl_label = icl_sample['output']

                if prompt_format == 'simple':

                    imp_prompts = [
                        f'Given [{icl_premise}] tell me if you can assume that [{icl_hypothesis}].\nIs it true, false or undetermined?\n\n{icl_label}\n\nGiven [{premise}] tell me if you can assume that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise}] tell me if you can infer that [{icl_hypothesis}].\nIs it true, false or undetermined?\n\n{icl_label}\n\nGiven [{premise}] tell me if you can infer that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise}] tell me if you can imply that [{icl_hypothesis}].\nIs it true, false or undetermined?\n\n{icl_label}\n\nGiven [{premise}] tell me if you can imply that [{hypothesis}].\nIs it true, false or undetermined?'
                    ]

                elif prompt_format == 'qa':

                    imp_prompts = [
                        f'Question: Given [{icl_premise}] tell me if you can assume that [{icl_hypothesis}].\nIs it true, false or undetermined?\nAnswer: {icl_label}\n\nQuestion: Given [{premise}] tell me if you can assume that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise}] tell me if you can infer that [{icl_hypothesis}].\nIs it true, false or undetermined?\nAnswer: {icl_label}\n\nQuestion: Given [{premise}] tell me if you can infer that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise}] tell me if you can imply that [{icl_hypothesis}].\nIs it true, false or undetermined?\nAnswer: {icl_label}\n\nQuestion: Given [{premise}] tell me if you can imply that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:'
                    ]

                else:
                    raise ValueError("Prompt format not supported")

            elif mode == "3shot":

                random.seed(0)
                icl_sample_1 = random.choice(testset)
                icl_premise_1 = icl_sample_1['premise']
                icl_hypothesis_1 = icl_sample_1['hypothesis']
                icl_label_1 = icl_sample_1['output']

                random.seed(1)
                icl_sample_2 = random.choice(testset)
                icl_premise_2 = icl_sample_2['premise']
                icl_hypothesis_2 = icl_sample_2['hypothesis']
                icl_label_2 = icl_sample_2['output']

                random.seed(4)
                icl_sample_3 = random.choice(testset)
                icl_premise_3 = icl_sample_3['premise']
                icl_hypothesis_3 = icl_sample_3['hypothesis']
                icl_label_3 = icl_sample_3['output']

                if prompt_format == 'simple':

                    imp_prompts = [
                        f'Given [{icl_premise_1}] tell me if you can assume that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\n\n{icl_label_1}\n\nGiven [{icl_premise_2}] tell me if you can assume that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\n\n{icl_label_2}\n\nGiven [{icl_premise_3}] tell me if you can assume that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\n\n{icl_label_3}\n\nGiven [{premise}] tell me if you can assume that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise_1}] tell me if you can infer that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\n\n{icl_label_1}\n\nGiven [{icl_premise_2}] tell me if you can infer that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\n\n{icl_label_2}\n\nGiven [{icl_premise_3}] tell me if you can infer that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\n\n{icl_label_3}\n\nGiven [{premise}] tell me if you can infer that [{hypothesis}].\nIs it true, false or undetermined?',
                        f'Given [{icl_premise_1}] tell me if you can imply that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\n\n{icl_label_1}\n\nGiven [{icl_premise_2}] tell me if you can imply that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\n\n{icl_label_2}\n\nGiven [{icl_premise_3}] tell me if you can imply that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\n\n{icl_label_3}\n\nGiven [{premise}] tell me if you can imply that [{hypothesis}].\nIs it true, false or undetermined?',
                    ]

                elif prompt_format == 'qa':

                    imp_prompts = [
                        f'Question: Given [{icl_premise_1}] tell me if you can assume that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\nAnswer: {icl_label_1}\n\nQuestion: Given [{icl_premise_2}] tell me if you can assume that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\nAnswer: {icl_label_2}\n\nQuestion: Given [{icl_premise_3}] tell me if you can assume that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\nAnswer: {icl_label_3}\n\nQuestion: Given [{premise}] tell me if you can assume that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise_1}] tell me if you can infer that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\nAnswer: {icl_label_1}\n\nQuestion: Given [{icl_premise_2}] tell me if you can infer that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\nAnswer: {icl_label_2}\n\nQuestion: Given [{icl_premise_3}] tell me if you can infer that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\nAnswer: {icl_label_3}\n\nQuestion: Given [{premise}] tell me if you can infer that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                        f'Question: Given [{icl_premise_1}] tell me if you can imply that [{icl_hypothesis_1}].\nIs it true, false or undetermined?\nAnswer: {icl_label_1}\n\nQuestion: Given [{icl_premise_2}] tell me if you can imply that [{icl_hypothesis_2}].\nIs it true, false or undetermined?\nAnswer: {icl_label_2}\n\nQuestion: Given [{icl_premise_3}] tell me if you can imply that [{icl_hypothesis_3}].\nIs it true, false or undetermined?\nAnswer: {icl_label_3}\n\nQuestion: Given [{premise}] tell me if you can imply that [{hypothesis}].\nIs it true, false or undetermined?\nAnswer:',
                    ]

                else:
                    raise ValueError("Prompt format not supported")

            for j, imp_prompt in enumerate(imp_prompts):
                
                sum = f"Prompt {j}\n"
                f.write(sum)

                single_turn_prompt = f"GPT4 Correct User: {imp_prompt}<|end_of_turn|>GPT4 Correct Assistant:"
                text_output = generate_response(model, tokenizer, single_turn_prompt)

                summary = f"\n{text_output}\n\n{label}\n"
                
                f.write(summary)
                f.write("-\n")

            print(f"Sample {i} done")
            f.write("---\n")


if __name__ == "__main__":

    model = AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5", torch_dtype = torch.float16) # if there is an error, change torch.float16 to torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5")
    model = model.to('cuda')
    mode = "0shot" # switch to 0shot, 1shot, 3shot
    prompt_format = "simple" # switch to simple, qa

    print("Starting passive inference...")
    passive_inference(model, tokenizer, mode, prompt_format)
    print("Passive inference done")
    print("Starting interrogative/active inference...")
    inter_active_inference(model, tokenizer, mode, prompt_format)
    print("Interrogative/active inference done")
    print("Starting indicative inference...")
    ind_inference(model, tokenizer, mode, prompt_format)
    print("Indicative inference done")
    print("Starting imperative inference...")
    imp_inference(model, tokenizer, mode, prompt_format)
    print("Imperative inference done")




