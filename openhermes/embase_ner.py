import random
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM    


class EmbaseNER(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./Embase_NER_test.json')
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

def passive_inference(model, tokenizer, mode):

    random.seed(0)
    testset = EmbaseNER().data
    random.shuffle(testset)
    testset = testset[:200]
    filename = f"./Embase_NER_passive_{mode}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):
            
            text = sample['input']
            label = sample['output']
            entity_type = sample['entity_type']
            structured_output = sample['structured_output']
            instruction_output_string = f"{{'{entity_type}' : [{entity_type}1, ..., {entity_type}N]}}"

            if mode == "0shot":

                passive_prompts = [
                    f'Question: Could the {entity_type}s be given from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Could the {entity_type}s be inferred from text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Could the {entity_type}s be found in the text.\n\nText: {text}\n\nAnswer:',
                ]

            elif mode == "1shot":
                                
                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_text = icl_sample1['input']
                icl_entity_type = icl_sample1['entity_type']
                icl_structured_output = icl_sample1['structured_output']

                passive_prompts = [
                    f'Question: Could the {icl_entity_type}s be given from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could the {entity_type}s be given from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Could the {icl_entity_type}s be inferred from text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could the {entity_type}s be inferred from text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Could the {icl_entity_type}s be found in the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could the {entity_type}s be found in the text.\n\nText: {text}\n\nAnswer:',
                ]

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

                passive_prompts = [
                    f'Question: Could the {icl_entity_type1}s be given from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could the {icl_entity_type2}s be given from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could the {icl_entity_type3}s be given from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could the {entity_type}s be given from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Could the {icl_entity_type1}s be inferred from text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could the {icl_entity_type2}s be inferred from text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could the {icl_entity_type3}s be inferred from text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could the {entity_type}s be inferred from text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Could the {icl_entity_type1}s be found in the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could the {icl_entity_type2}s be found in the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could the {icl_entity_type3}s be found in the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could the {entity_type}s be found in the text.\n\nText: {text}\n\nAnswer:',
                ]

            else:
                raise ValueError("Invalid mode")

            for j, passive_prompt in enumerate(passive_prompts):
                
                sum = f"Prompt {j}\n"
                f.write(sum)

                messages = [
                    {"role": "system", "content": f"You are a Named Entity Recognition expert. Your answers must have the format: {instruction_output_string}"},
                    {"role": "user", "content": passive_prompt},
                ]
                gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                generated_ids = model.generate(gen_input, max_new_tokens=200, do_sample=True)
                decoded = tokenizer.batch_decode(generated_ids)
                text_output = decoded[0]

                summary = f"\n{text_output}\n\n{label}\n"
                
                f.write(summary)
                f.write("-\n")

            print(f"Sample {i} done")
            f.write("---\n")


def inter_active_inference(model, tokenizer, mode):

    random.seed(0)
    testset = EmbaseNER().data
    random.shuffle(testset)
    testset = testset[:200]

    filename = f"./Embase_NER_inter_active_{mode}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):
            
            text = sample['input']
            label = sample['output']
            entity_type = sample['entity_type']
            structured_output = sample['structured_output']
            instruction_output_string = f"{{'{entity_type}' : [{entity_type}1, ..., {entity_type}N]}}"

            if mode == "0shot":

                inter_active_prompts = [
                    f'Question: Could you give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Could you infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Could you find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                ]

            elif mode == "1shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_text = icl_sample1['input']
                icl_entity_type = icl_sample1['entity_type']
                icl_structured_output = icl_sample1['structured_output']

                inter_active_prompts = [
                    f'Question: Could you give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could you give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Could you infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could you infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Could you find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Could you find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                ]

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

                inter_active_prompts = [
                    f'Question: Could you give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could you give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could you give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could you give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Could you infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could you infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could you infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could you infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Could you find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Could you find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Could you find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Could you find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                ]

            for j, inter_active_prompt in enumerate(inter_active_prompts):
                
                sum = f"Prompt {j}\n"
                f.write(sum)

                messages = [
                    {"role": "system", "content": f"You are a Named Entity Recognition expert. Your answers must have the format: {instruction_output_string}"},
                    {"role": "user", "content": inter_active_prompt},
                ]
                gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                generated_ids = model.generate(gen_input, max_new_tokens=200, do_sample=True)
                decoded = tokenizer.batch_decode(generated_ids)
                text_output = decoded[0]

                summary = f"\n{text_output}\n\n{label}\n"
                
                f.write(summary)
                f.write("-\n")

            print(f"Sample {i} done")
            f.write("---\n")


def ind_inference(model, tokenizer, mode):

    random.seed(0)
    testset = EmbaseNER().data
    random.shuffle(testset)
    testset = testset[:200]

    filename = f"./Embase_NER_ind_{mode}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):
            
            text = sample['input']
            label = sample['output']
            entity_type = sample['entity_type']
            structured_output = sample['structured_output']
            instruction_output_string = f"{{'{entity_type}' : [{entity_type}1, ..., {entity_type}N]}}"

            if mode == "0shot":

                ind_prompts = [
                    f'Question: You give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: You infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: You find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                ]

            elif mode == "1shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_text = icl_sample1['input']
                icl_entity_type = icl_sample1['entity_type']
                icl_structured_output = icl_sample1['structured_output']

                ind_prompts = [
                    f'Question: You give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: You give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: You infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: You infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: You find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: You find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                ]

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

                ind_prompts = [
                    f'Question: You give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: You give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: You give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: You give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: You infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: You infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: You infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: You infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: You find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: You find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: You find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: You find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                ]

            for j, ind_prompt in enumerate(ind_prompts):
                
                sum = f"Prompt {j}\n"
                f.write(sum)

                messages = [
                    {"role": "system", "content": f"You are a Named Entity Recognition expert. Your answers must have the format: {instruction_output_string}"},
                    {"role": "user", "content": ind_prompt},
                ]
                gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                generated_ids = model.generate(gen_input, max_new_tokens=200, do_sample=True)
                decoded = tokenizer.batch_decode(generated_ids)
                text_output = decoded[0]

                summary = f"\n{text_output}\n\n{label}\n"
                
                f.write(summary)
                f.write("-\n")

            print(f"Sample {i} done")
            f.write("---\n")


def imp_inference(model, tokenizer, mode):

    random.seed(0)
    testset = EmbaseNER().data
    random.shuffle(testset)
    testset = testset[:200]

    filename = f"./Embase_NER_imp_{mode}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):
            
            text = sample['input']
            label = sample['output']
            entity_type = sample['entity_type']
            structured_output = sample['structured_output']
            instruction_output_string = f"{{'{entity_type}' : [{entity_type}1, ..., {entity_type}N]}}"

            if mode == "0shot":

                imp_prompts = [
                    f'Question: Give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                ]
            
            elif mode == "1shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_text = icl_sample1['input']
                icl_entity_type = icl_sample1['entity_type']
                icl_structured_output = icl_sample1['structured_output']

                imp_prompts = [
                    f'Question: Give me the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Infer the {icl_entity_type}s from the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Find the {icl_entity_type}s in the text.\n\nText: {icl_text}\n\nAnswer: {icl_structured_output}\n\nQuestion: Find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                ]
            
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

                imp_prompts = [
                    f'Question: Give me the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Give me the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Give me the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Give me the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Infer the {icl_entity_type1}s from the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Infer the {icl_entity_type2}s from the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Infer the {icl_entity_type3}s from the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Infer the {entity_type}s from the text.\n\nText: {text}\n\nAnswer:',
                    f'Question: Find the {icl_entity_type1}s in the text.\n\nText: {icl_text1}\n\nAnswer: {icl_structured_output1}\n\nQuestion: Find the {icl_entity_type2}s in the text.\n\nText: {icl_text2}\n\nAnswer: {icl_structured_output2}\n\nQuestion: Find the {icl_entity_type3}s in the text.\n\nText: {icl_text3}\n\nAnswer: {icl_structured_output3}\n\nQuestion: Find the {entity_type}s in the text.\n\nText: {text}\n\nAnswer:',
                ]

            for j, imp_prompt in enumerate(imp_prompts):
                
                sum = f"Prompt {j}\n"
                f.write(sum)

                messages = [
                    {"role": "system", "content": f"You are a Named Entity Recognition expert. Your answers must have the format: {instruction_output_string}"},
                    {"role": "user", "content": imp_prompt},
                ]
                gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                generated_ids = model.generate(gen_input, max_new_tokens=200, do_sample=True)
                decoded = tokenizer.batch_decode(generated_ids)
                text_output = decoded[0]

                summary = f"\n{text_output}\n\n{label}\n"
                
                f.write(summary)
                f.write("-\n")

            print(f"Sample {i} done")
            f.write("---\n")


if __name__ == "__main__":
    
    model = AutoTokenizer.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B", torch_dtype = torch.float16)
    tokenizer = AutoModelForCausalLM.from_pretrained("teknium/OpenHermes-2.5-Mistral-7B")
    model = model.to("cuda")
    mode = "0shot"

    print("Starting passive inference...")
    passive_inference(model, tokenizer, mode)
    print("Passive inference done")
    print("Starting interrogative/active inference...")
    inter_active_inference(model, tokenizer, mode)
    print("Interrogative/active inference done")
    print("Starting indicative inference...")
    ind_inference(model, tokenizer, mode)
    print("Indicative inference done")
    print("Starting imperative inference...")
    imp_inference(model, tokenizer, mode)
    print("Imperative inference done")
