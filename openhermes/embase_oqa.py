import random
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM    


class EmbaseOQA(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.data = self.read_and_filter('./Embase_OpenBookQA_test.json')
        self.data = [{'instruction': entry['instruction'], 'context': entry['input']['context'], 'question': entry['input']['question'], 'output': entry['output']} for entry in self.data]

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
            self.has_special_characters(item['input']['context']) or
            self.has_special_characters(item['input']['question'])
        )]

        return filtered_list


def passive_inference(model, tokenizer, mode):

    random.seed(0)
    testset = EmbaseOQA().data

    filename = f"./Embase_OQA_passive_{mode}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):

            question = sample['question']
            context = sample['context']
            label = sample['output']

            if mode == "0shot":

                passive_prompts = [
                    f"What is it thought to be the correct answer based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                    f"Could the correct answer be inferred based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                    f"Could the correct answer be given based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                    f"Could the question be solved by answering based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                ]

            elif mode == "1shot":

                random.seed(14)
                icl_sample = random.choice(testset)
                icl_question = icl_sample['question']
                icl_context = icl_sample['context']
                icl_label = icl_sample['output']

                passive_prompts = [
                    f"What is it thought to be the correct answer based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nWhat is it thought to be the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Could the correct answer be inferred based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nCould the correct answer be inferred based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Could the correct answer be given based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nCould the correct answer be given based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Could the question be solved by answering based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nCould the question be solved by answering based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                ]

            elif mode == "3shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_question1 = icl_sample1['question']
                icl_context1 = icl_sample1['context']
                icl_label1 = icl_sample1['output']

                random.seed(14)
                icl_sample2 = random.choice(testset)
                icl_question2 = icl_sample2['question']
                icl_context2 = icl_sample2['context']
                icl_label2 = icl_sample2['output']

                random.seed(17)
                icl_sample3 = random.choice(testset)
                icl_question3 = icl_sample3['question']
                icl_context3 = icl_sample3['context']
                icl_label3 = icl_sample3['output']

                passive_prompts = [
                    f"What is it thought to be the correct answer based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nWhat is it thought to be the correct answer based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nWhat is it thought to be the correct answer based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nWhat is it thought to be the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Could the correct answer be inferred based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nCould the correct answer be inferred based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nCould the correct answer be inferred based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nCould the correct answer be inferred based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Could the correct answer be given based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nCould the correct answer be given based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nCould the correct answer be given based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nCould the correct answer be given based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Could the question be solved by answering based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nCould the question be solved by answering based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nCould the question be solved by answering based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nCould the question be solved by answering based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                ]

            else:
                raise ValueError("Invalid mode")

            for j, passive_prompt in enumerate(passive_prompts):
                
                sum = f"Prompt {j}\n"
                f.write(sum)

                messages = [
                    {"role": "system", "content": "You are a Question Answering expert. Your answers must be as short as possible and concise."},
                    {"role": "user", "content": passive_prompt},
                ]
                gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                generated_ids = model.generate(gen_input, max_new_tokens=300, do_sample=True)
                decoded = tokenizer.batch_decode(generated_ids)
                text_output = decoded[0]

                summary = f"\n{text_output}\n\n{label}\n"
                
                f.write(summary)
                f.write("-\n")

            print(f"Sample {i} done")
            f.write("---\n")


def inter_active_inference(model, tokenizer, mode):

    random.seed(0)
    testset = EmbaseOQA().data

    filename = f"./Embase_OQA_inter_active_{mode}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):

            question = sample['question']
            context = sample['context']
            label = sample['output']

            if mode == "0shot":

                inter_active_prompts = [
                    f"What do you think is the correct answer based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                    f"Could you infer the correct answer based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                    f"Could you give me the correct answer based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                    f"Could you solve the question by answering based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                ]
            
            elif mode == "1shot":
                
                random.seed(14)
                icl_sample = random.choice(testset)
                icl_question = icl_sample['question']
                icl_context = icl_sample['context']
                icl_label = icl_sample['output']

                inter_active_prompts = [
                    f"What do you think is the correct answer based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nWhat do you think is the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Could you infer the correct answer based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nCould you infer the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Could you give me the correct answer based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nCould you give me the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Could you solve the question by answering based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nCould you solve the question by answering based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                ]

            elif mode == "3shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_question1 = icl_sample1['question']
                icl_context1 = icl_sample1['context']
                icl_label1 = icl_sample1['output']

                random.seed(14)
                icl_sample2 = random.choice(testset)
                icl_question2 = icl_sample2['question']
                icl_context2 = icl_sample2['context']
                icl_label2 = icl_sample2['output']

                random.seed(17)
                icl_sample3 = random.choice(testset)
                icl_question3 = icl_sample3['question']
                icl_context3 = icl_sample3['context']
                icl_label3 = icl_sample3['output']

                inter_active_prompts = [
                    f"What do you think is the correct answer based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nWhat do you think is the correct answer based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nWhat do you think is the correct answer based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nWhat do you think is the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Could you infer the correct answer based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nCould you infer the correct answer based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nCould you infer the correct answer based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nCould you infer the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Could you give me the correct answer based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nCould you give me the correct answer based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nCould you give me the correct answer based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nCould you give me the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Could you solve the question by answering based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nCould you solve the question by answering based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nCould you solve the question by answering based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nCould you solve the question by answering based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                ]

            else:
                raise ValueError("Invalid mode")

            for j, inter_active_prompt in enumerate(inter_active_prompts):
                
                sum = f"Prompt {j}\n"
                f.write(sum)

                messages = [
                    {"role": "system", "content": "You are a Question Answering expert. Your answers must be short and concise."},
                    {"role": "user", "content": inter_active_prompt},
                ]
                gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                generated_ids = model.generate(gen_input, max_new_tokens=300, do_sample=True)
                decoded = tokenizer.batch_decode(generated_ids)
                text_output = decoded[0]

                summary = f"\n{text_output}\n\n{label}\n"
                
                f.write(summary)
                f.write("-\n")

            print(f"Sample {i} done")
            f.write("---\n")


def ind_inference(model, tokenizer, mode):

    random.seed(0)
    testset = EmbaseOQA().data

    filename = f"./Embase_OQA_ind_{mode}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):

            question = sample['question']
            context = sample['context']
            label = sample['output']

            if mode == "0shot":

                ind_prompts = [
                    f"You infer the correct answer based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                    f"You give me the correct answer based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                    f"You solve the question by answering based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                ]
            
            elif mode == "1shot":

                random.seed(14)
                icl_sample = random.choice(testset)
                icl_question = icl_sample['question']
                icl_context = icl_sample['context']
                icl_label = icl_sample['output']

                ind_prompts = [
                    f"You infer the correct answer based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nYou infer the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"You give me the correct answer based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nYou give me the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"You solve the question by answering based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nYou solve the question by answering based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                ]

            elif mode == "3shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_question1 = icl_sample1['question']
                icl_context1 = icl_sample1['context']
                icl_label1 = icl_sample1['output']

                random.seed(14)
                icl_sample2 = random.choice(testset)
                icl_question2 = icl_sample2['question']
                icl_context2 = icl_sample2['context']
                icl_label2 = icl_sample2['output']

                random.seed(17)
                icl_sample3 = random.choice(testset)
                icl_question3 = icl_sample3['question']
                icl_context3 = icl_sample3['context']
                icl_label3 = icl_sample3['output']

                ind_prompts = [
                    f"You infer the correct answer based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nYou infer the correct answer based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nYou infer the correct answer based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nYou infer the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"You give me the correct answer based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nYou give me the correct answer based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nYou give me the correct answer based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nYou give me the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"You solve the question by answering based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nYou solve the question by answering based on the context.\nQuestion {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nYou solve the question by answering based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nYou solve the question by answering based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                ]

            else:
                raise ValueError("Invalid mode")

            for j, ind_prompt in enumerate(ind_prompts):
                
                sum = f"Prompt {j}\n"
                f.write(sum)

                messages = [
                    {"role": "system", "content": "You are a Question Answering expert. Your answers must be as short as possible and concise."},
                    {"role": "user", "content": ind_prompt},
                ]
                gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                generated_ids = model.generate(gen_input, max_new_tokens=300, do_sample=True)
                decoded = tokenizer.batch_decode(generated_ids)
                text_output = decoded[0]

                summary = f"\n{text_output}\n\n{label}\n"
                
                f.write(summary)
                f.write("-\n")

            print(f"Sample {i} done")
            f.write("---\n")


def imp_inference(model, tokenizer, mode):

    random.seed(0)
    testset = EmbaseOQA().data

    filename = f"./Embase_OQA_imp_{mode}.txt"

    with open(filename, 'w') as f:
        
        for i,sample in enumerate(testset):

            question = sample['question']
            context = sample['context']
            label = sample['output']

            if mode == '0shot':

                imp_prompts = [
                    f"Infer the correct answer based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                    f"Give me the correct answer based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                    f"Solve the question by answering based on the context.\nQuestion:{question}\nContext:{context}\nAnswer:",
                ]

            elif mode == "1shot":

                random.seed(14)
                icl_sample = random.choice(testset)
                icl_question = icl_sample['question']
                icl_context = icl_sample['context']
                icl_label = icl_sample['output']

                imp_prompts = [
                    f"Infer the correct answer based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nInfer the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Give me the correct answer based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nGive me the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Solve the question by answering based on the context.\nQuestion: {icl_question}\nContext: {icl_context}\nAnswer: {icl_label}\n\nSolve the question by answering based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                ]

            elif mode == "3shot":

                random.seed(1)
                icl_sample1 = random.choice(testset)
                icl_question1 = icl_sample1['question']
                icl_context1 = icl_sample1['context']
                icl_label1 = icl_sample1['output']

                random.seed(14)
                icl_sample2 = random.choice(testset)
                icl_question2 = icl_sample2['question']
                icl_context2 = icl_sample2['context']
                icl_label2 = icl_sample2['output']

                random.seed(17)
                icl_sample3 = random.choice(testset)
                icl_question3 = icl_sample3['question']
                icl_context3 = icl_sample3['context']
                icl_label3 = icl_sample3['output']

                imp_prompts = [
                    f"Infer the correct answer based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nInfer the correct answer based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nInfer the correct answer based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nInfer the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Give me the correct answer based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nGive me the correct answer based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nGive me the correct answer based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nGive me the correct answer based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                    f"Solve the question by answering based on the context.\nQuestion: {icl_question1}\nContext: {icl_context1}\nAnswer: {icl_label1}\n\nSolve the question by answering based on the context.\nQuestion: {icl_question2}\nContext: {icl_context2}\nAnswer: {icl_label2}\n\nSolve the question by answering based on the context.\nQuestion: {icl_question3}\nContext: {icl_context3}\nAnswer: {icl_label3}\n\nSolve the question by answering based on the context.\nQuestion: {question}\nContext: {context}\nAnswer:",
                ]

            else:
                raise ValueError("Invalid mode")

            for j, imp_prompt in enumerate(imp_prompts):
                
                sum = f"Prompt {j}\n"
                f.write(sum)

                messages = [
                    {"role": "system", "content": "You are a Question Answering expert. Your answers must be as short as possible and concise."},
                    {"role": "user", "content": imp_prompt},
                ]
                gen_input = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
                generated_ids = model.generate(gen_input, max_new_tokens=300, do_sample=True)
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


