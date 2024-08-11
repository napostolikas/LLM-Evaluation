import random
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM    

def generate_response(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to('cuda')
    outputs = model.generate(
        input_ids,
        max_new_tokens=300,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = outputs[0]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text

class EmbaseNER(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./Embase_NER_samples.json')
        self.data = [{'instruction': entry['instruction'], 'input': entry['input'], 'output': entry['output'], 'entity_type': entry['entity_type'], 'structured_output': entry['structured_output']} for entry in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def has_special_characters(self, text):
        return any(ord(char) > 127 for char in text)

    def read_and_filter(self, file_name: str) -> list:
        with open(file_name, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        filtered_list = [item for item in data if not self.has_special_characters(item['input'])]
        return filtered_list

def passive_inference(model, tokenizer, mode, prompt_format, provide_format):

    random.seed(0)
    testset = EmbaseNER().data
    random.shuffle(testset)
    testset = testset[:200]
    filename = f"./Embase_NER_passive_{mode}_{prompt_format}_{provide_format}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):
            
            text = sample['input']
            label = sample['output']
            entity_type = sample['entity_type']
            structured_output = sample['structured_output']
            instruction_output_string = f"{{'{entity_type}' : [{entity_type}1, ..., {entity_type}N]}}"

            if mode == "0shot":

                if prompt_format == 'qa':

                    passive_prompts = [
                        f'Question: Could the {entity_type}s be given from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        f'Question: Could the {entity_type}s be inferred from text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        f'Question: Could the {entity_type}s be found in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                    ]

                elif prompt_format == 'simple':

                    passive_prompts = [
                        f'Could the {entity_type}s be given from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        f'Could the {entity_type}s be inferred from text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        f'Could the {entity_type}s be found in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                    ]                

            elif mode == "1shot":
                                
                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_text = icl_sample1['input']
                icl_entity_type = icl_sample1['entity_type']
                icl_structured_output = icl_sample1['structured_output']

                if prompt_format == 'qa':

                    if provide_format:

                        passive_prompts = [
                            f'Question: Could the {icl_entity_type}s be given from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could the {entity_type}s be given from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: Could the {icl_entity_type}s be inferred from text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could the {entity_type}s be inferred from text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: Could the {icl_entity_type}s be found in the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could the {entity_type}s be found in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        ]

                    else:

                        passive_prompts = [
                            f'Question: Could the {icl_entity_type}s be given from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could the {entity_type}s be given from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: Could the {icl_entity_type}s be inferred from text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could the {entity_type}s be inferred from text.\n\nText: {text}\n\nAnswer:',
                            f'Question: Could the {icl_entity_type}s be found in the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could the {entity_type}s be found in the text.\n\nText: {text}\n\nAnswer:',
                        ]

                elif prompt_format == 'simple':

                    if provide_format:

                        passive_prompts = [
                            f'Could the {icl_entity_type}s be given from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nCould the {entity_type}s be given from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'Could the {icl_entity_type}s be inferred from text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nCould the {entity_type}s be inferred from text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'Could the {icl_entity_type}s be found in the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nCould the {entity_type}s be found in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        ]
                    
                    else:

                        passive_prompts = [
                            f'Could the {icl_entity_type}s be given from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nCould the {entity_type}s be given from the text.\n\nText: {text}',
                            f'Could the {icl_entity_type}s be inferred from text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nCould the {entity_type}s be inferred from text.\n\nText: {text}',
                            f'Could the {icl_entity_type}s be found in the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nCould the {entity_type}s be found in the text.\n\nText: {text}',
                        ]
                
                else:
                    raise ValueError("Invalid prompt format")

            elif mode == "3shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_text1 = icl_sample1['input']
                icl_entity_type1 = icl_sample1['entity_type']
                icl_structured_output1 = icl_sample1['structured_output']

                random.seed(4)
                icl_sample2 = random.choice(testset)
                icl_text2 = icl_sample2['input']
                icl_entity_type2 = icl_sample2['entity_type']
                icl_structured_output2 = icl_sample2['structured_output']

                random.seed(6)
                icl_sample3 = random.choice(testset)
                icl_text3 = icl_sample3['input']
                icl_entity_type3 = icl_sample3['entity_type']
                icl_structured_output3 = icl_sample3['structured_output']

                if prompt_format == 'qa':

                    if provide_format:

                        passive_prompts = [
                            f'Question: Could the {icl_entity_type1}s be given from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could the {icl_entity_type2}s be given from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could the {icl_entity_type3}s be given from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could the {entity_type}s be given from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: Could the {icl_entity_type1}s be inferred from text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could the {icl_entity_type2}s be inferred from text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could the {icl_entity_type3}s be inferred from text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could the {entity_type}s be inferred from text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: Could the {icl_entity_type1}s be found in the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could the {icl_entity_type2}s be found in the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could the {icl_entity_type3}s be found in the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could the {entity_type}s be found in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        ]

                    else:

                        passive_prompts = [
                            f'Question: Could the {icl_entity_type1}s be given from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could the {icl_entity_type2}s be given from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could the {icl_entity_type3}s be given from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could the {entity_type}s be given from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: Could the {icl_entity_type1}s be inferred from text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could the {icl_entity_type2}s be inferred from text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could the {icl_entity_type3}s be inferred from text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could the {entity_type}s be inferred from text.\n\nText: {text}\n\nAnswer:',
                            f'Question: Could the {icl_entity_type1}s be found in the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could the {icl_entity_type2}s be found in the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could the {icl_entity_type3}s be found in the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could the {entity_type}s be found in the text.\n\nText: {text}\n\nAnswer:',
                        ]
                
                elif prompt_format == 'simple':

                    if provide_format:

                        passive_prompts = [
                            f'Could the {icl_entity_type1}s be given from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nCould the {icl_entity_type2}s be given from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nCould the {icl_entity_type3}s be given from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nCould the {entity_type}s be given from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'Could the {icl_entity_type1}s be inferred from text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nCould the {icl_entity_type2}s be inferred from text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nCould the {icl_entity_type3}s be inferred from text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nCould the {entity_type}s be inferred from text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'Could the {icl_entity_type1}s be found in the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nCould the {icl_entity_type2}s be found in the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nCould the {icl_entity_type3}s be found in the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nCould the {entity_type}s be found in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        ]

                    else:

                        passive_prompts = [
                            f'Could the {icl_entity_type1}s be given from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nCould the {icl_entity_type2}s be given from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nCould the {icl_entity_type3}s be given from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nCould the {entity_type}s be given from the text.\n\nText: {text}',
                            f'Could the {icl_entity_type1}s be inferred from text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nCould the {icl_entity_type2}s be inferred from text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nCould the {icl_entity_type3}s be inferred from text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nCould the {entity_type}s be inferred from text.\n\nText: {text}',
                            f'Could the {icl_entity_type1}s be found in the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nCould the {icl_entity_type2}s be found in the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nCould the {icl_entity_type3}s be found in the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nCould the {entity_type}s be found in the text.\n\nText: {text}',
                        ]
                
                else:
                    raise ValueError("Invalid prompt format")
                
            else:
                raise ValueError("Invalid mode")

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


def inter_active_inference(model, tokenizer, mode, prompt_format, provide_format):

    random.seed(0)
    testset = EmbaseNER().data
    random.shuffle(testset)
    testset = testset[:200]

    filename = f"./Embase_NER_inter_active_{mode}_{prompt_format}_{provide_format}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):
            
            text = sample['input']
            label = sample['output']
            entity_type = sample['entity_type']
            structured_output = sample['structured_output']
            instruction_output_string = f"{{'{entity_type}' : [{entity_type}1, ..., {entity_type}N]}}"

            if mode == "0shot":

                if prompt_format == 'qa':

                    inter_active_prompts = [
                        f'Question: Could you give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        f'Question: Could you infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        f'Question: Could you find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                    ]

                elif prompt_format == 'simple':

                    inter_active_prompts = [
                        f'Could you give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        f'Could you infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        f'Could you find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                    ]
                
                else:
                    raise ValueError("Invalid prompt format")

            elif mode == "1shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_text = icl_sample1['input']
                icl_entity_type = icl_sample1['entity_type']
                icl_structured_output = icl_sample1['structured_output']

                if prompt_format == 'qa':

                    if provide_format:

                        inter_active_prompts = [
                            f'Question: Could you give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could you give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: Could you infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could you infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: Could you find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could you find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        ]

                    else:

                        inter_active_prompts = [
                            f'Question: Could you give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could you give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: Could you infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could you infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: Could you find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could you find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                        ]
                    
                elif prompt_format == 'simple':

                    if provide_format:

                        inter_active_prompts = [
                            f'Could you give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nCould you give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'Could you infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nCould you infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'Could you find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nCould you find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        ]

                    else:

                        inter_active_prompts = [
                            f'Could you give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nCould you give me the {entity_type}s from the text.\n\nText: {text}',
                            f'Could you infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nCould you infer the {entity_type}s from the text.\n\nText: {text}',
                            f'Could you find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nCould you find the {entity_type}s in the text.\n\nText: {text}',
                        ]
                
                else:
                    raise ValueError("Invalid prompt format")

            elif mode == "3shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_text1 = icl_sample1['input']
                icl_entity_type1 = icl_sample1['entity_type']
                icl_structured_output1 = icl_sample1['structured_output']

                random.seed(4)
                icl_sample2 = random.choice(testset)
                icl_text2 = icl_sample2['input']
                icl_entity_type2 = icl_sample2['entity_type']
                icl_structured_output2 = icl_sample2['structured_output']

                random.seed(6)
                icl_sample3 = random.choice(testset)
                icl_text3 = icl_sample3['input']
                icl_entity_type3 = icl_sample3['entity_type']
                icl_structured_output3 = icl_sample3['structured_output']

                if prompt_format == 'qa':

                    if provide_format:

                        inter_active_prompts = [
                            f'Question: Could you give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could you give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could you give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could you give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: Could you infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could you infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could you infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could you infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: Could you find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could you find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could you find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could you find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        ]

                    else:

                        inter_active_prompts = [
                            f'Question: Could you give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could you give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could you give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could you give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: Could you infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could you infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could you infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could you infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: Could you find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could you find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could you find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could you find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                        ]
                    
                elif prompt_format == 'simple':

                    if provide_format:

                        inter_active_prompts = [
                            f'Could you give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nCould you give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nCould you give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nCould you give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'Could you infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nCould you infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nCould you infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nCould you infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'Could you find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nCould you find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nCould you find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nCould you find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        ]

                    else:

                        inter_active_prompts = [
                            f'Could you give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nCould you give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nCould you give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nCould you give me the {entity_type}s from the text.\n\nText: {text}',
                            f'Could you infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nCould you infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nCould you infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nCould you infer the {entity_type}s from the text.\n\nText: {text}',
                            f'Could you find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nCould you find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nCould you find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nCould you find the {entity_type}s in the text.\n\nText: {text}',
                        ]

                else:
                    raise ValueError("Invalid prompt format")

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


def ind_inference(model, tokenizer, mode, prompt_format, provide_format):

    random.seed(0)
    testset = EmbaseNER().data
    random.shuffle(testset)
    testset = testset[:200]

    filename = f"./Embase_NER_ind_{mode}_{prompt_format}_{provide_format}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):
            
            text = sample['input']
            label = sample['output']
            entity_type = sample['entity_type']
            structured_output = sample['structured_output']
            instruction_output_string = f"{{'{entity_type}' : [{entity_type}1, ..., {entity_type}N]}}"

            if mode == "0shot":
                
                if prompt_format == 'qa':

                    ind_prompts = [
                        f'Question: You give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        f'Question: You infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        f'Question: You find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                    ]

                elif prompt_format == 'simple':

                    ind_prompts = [
                        f'You give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        f'You infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        f'You find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                    ]

                else:
                    raise ValueError("Invalid prompt format")

            elif mode == "1shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_text = icl_sample1['input']
                icl_entity_type = icl_sample1['entity_type']
                icl_structured_output = icl_sample1['structured_output']

                if prompt_format == 'qa':
                    
                    if provide_format:

                        ind_prompts = [
                            f'Question: You give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: You give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: You infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: You infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: You find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: You find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        ]

                    else:

                        ind_prompts = [
                            f'Question: You give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: You give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: You infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: You infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: You find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: You find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                        ]
                
                elif prompt_format == 'simple':

                    if provide_format:

                        ind_prompts = [
                            f'You give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nYou give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'You infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nYou infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'You find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nYou find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        ]

                    else:

                        ind_prompts = [
                            f'You give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nYou give me the {entity_type}s from the text.\n\nText: {text}',
                            f'You infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nYou infer the {entity_type}s from the text.\n\nText: {text}',
                            f'You find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nYou find the {entity_type}s in the text.\n\nText: {text}',
                        ]
                
                else:
                    raise ValueError("Invalid prompt format")

            elif mode == "3shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_text1 = icl_sample1['input']
                icl_entity_type1 = icl_sample1['entity_type']
                icl_structured_output1 = icl_sample1['structured_output']

                random.seed(4)
                icl_sample2 = random.choice(testset)
                icl_text2 = icl_sample2['input']
                icl_entity_type2 = icl_sample2['entity_type']
                icl_structured_output2 = icl_sample2['structured_output']

                random.seed(6)
                icl_sample3 = random.choice(testset)
                icl_text3 = icl_sample3['input']
                icl_entity_type3 = icl_sample3['entity_type']
                icl_structured_output3 = icl_sample3['structured_output']

                if prompt_format == 'qa':

                    if provide_format:

                        ind_prompts = [
                            f'Question: You give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: You give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: You give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: You give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: You infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: You infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: You infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: You infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: You find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: You find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: You find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: You find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        ]

                    else:

                        ind_prompts = [
                            f'Question: You give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: You give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: You give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: You give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: You infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: You infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: You infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: You infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: You find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: You find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: You find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: You find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                        ]
                
                elif prompt_format == 'simple':

                    if provide_format:

                        ind_prompts = [
                            f'You give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nYou give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nYou give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nYou give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'You infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nYou infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nYou infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nYou infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'You find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nYou find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nYou find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nYou find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        ]

                    else:

                        ind_prompts = [
                            f'You give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nYou give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nYou give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nYou give me the {entity_type}s from the text.\n\nText: {text}',
                            f'You infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nYou infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nYou infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nYou infer the {entity_type}s from the text.\n\nText: {text}',
                            f'You find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nYou find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nYou find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nYou find the {entity_type}s in the text.\n\nText: {text}',
                        ]

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


def imp_inference(model, tokenizer, mode, prompt_format, provide_format):

    random.seed(0)
    testset = EmbaseNER().data
    random.shuffle(testset)
    testset = testset[:200]

    filename = f"./Embase_NER_imp_{mode}_{prompt_format}_{provide_format}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):
            
            text = sample['input']
            label = sample['output']
            entity_type = sample['entity_type']
            structured_output = sample['structured_output']
            instruction_output_string = f"{{'{entity_type}' : [{entity_type}1, ..., {entity_type}N]}}"

            if mode == "0shot":

                if prompt_format == 'qa':

                    imp_prompts = [
                        f'Question: Give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        f'Question: Infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        f'Question: Find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                    ]

                elif prompt_format == 'simple':

                    imp_prompts = [
                        f'Give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        f'Infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        f'Find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                    ]
            
            elif mode == "1shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_text = icl_sample1['input']
                icl_entity_type = icl_sample1['entity_type']
                icl_structured_output = icl_sample1['structured_output']

                if prompt_format == 'qa':

                    if provide_format:

                        imp_prompts = [
                            f'Question: Give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: Infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: Find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        ]
                    
                    else:

                        imp_prompts = [
                            f'Question: Give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: Infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: Find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                        ]

                elif prompt_format == 'simple':

                    if provide_format:

                        imp_prompts = [
                            f'Give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nGive me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'Infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nInfer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'Find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nFind the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        ]

                    else:

                        imp_prompts = [
                            f'Give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nGive me the {entity_type}s from the text.\n\nText: {text}',
                            f'Infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nInfer the {entity_type}s from the text.\n\nText: {text}',
                            f'Find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\n{icl_structured_output}\n\nFind the {entity_type}s in the text.\n\nText: {text}',
                        ]

                else:
                    raise ValueError("Invalid prompt format")
            
            elif mode == "3shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_text1 = icl_sample1['input']
                icl_entity_type1 = icl_sample1['entity_type']
                icl_structured_output1 = icl_sample1['structured_output']

                random.seed(4)
                icl_sample2 = random.choice(testset)
                icl_text2 = icl_sample2['input']
                icl_entity_type2 = icl_sample2['entity_type']
                icl_structured_output2 = icl_sample2['structured_output']

                random.seed(6)
                icl_sample3 = random.choice(testset)
                icl_text3 = icl_sample3['input']
                icl_entity_type3 = icl_sample3['entity_type']
                icl_structured_output3 = icl_sample3['structured_output']

                if prompt_format == 'qa':

                    if provide_format:

                        imp_prompts = [
                            f'Question: Give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Give me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: Infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Infer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                            f'Question: Find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Find the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}\n\nAnswer:',
                        ]

                    else:

                        imp_prompts = [
                            f'Question: Give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: Infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                            f'Question: Find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                        ]

                elif prompt_format == 'simple':

                    if provide_format:

                        imp_prompts = [
                            f'Give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nGive me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nGive me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nGive me the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'Infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nInfer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nInfer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nInfer the {entity_type}s from the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                            f'Find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nFind the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nFind the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nFind the {entity_type}s in the text.\n\nText: {text}\n\nOutput format: {instruction_output_string}',
                        ]

                    else:

                        imp_prompts = [
                            f'Give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nGive me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nGive me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nGive me the {entity_type}s from the text.\n\nText: {text}',
                            f'Infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nInfer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nInfer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nInfer the {entity_type}s from the text.\n\nText: {text}',
                            f'Find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\n{icl_structured_output1}\n\nFind the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\n{icl_structured_output2}\n\nFind the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\n{icl_structured_output3}\n\nFind the {entity_type}s in the text.\n\nText: {text}',
                        ]

                else:
                    raise ValueError("Invalid prompt format")

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
    provide_format = False

    print("Starting passive inference...")
    passive_inference(model, tokenizer, mode, prompt_format, provide_format)
    print("Passive inference done")
    print("Starting interrogative/active inference...")
    inter_active_inference(model, tokenizer, mode, prompt_format, provide_format)
    print("Interrogative/active inference done")
    print("Starting indicative inference...")
    ind_inference(model, tokenizer, mode, prompt_format, provide_format)
    print("Indicative inference done")
    print("Starting imperative inference...")
    imp_inference(model, tokenizer, mode, prompt_format, provide_format)
    print("Imperative inference done")
