import torch
import json
from nltk import PorterStemmer
from prettytable import PrettyTable

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
    

def extract_answers_qa(file_name: str) -> dict:

    with open(file_name, 'r') as file:
        content = file.read()

    prompts_dict = {}
    current_prompt = None

    for i,line in enumerate(content.split('\n')):
        line = line.strip()
        
        if line.startswith("Prompt"):
            current_prompt = line

        elif line.startswith("Answer:<|im_end|>"):

            if content.split('\n')[i+1] != "":
                answer_text = content.split('\n')[i+1].lower().replace(".","")
            else:
                answer_text = content.split('\n')[i+2].lower().replace(".","")

            if "<|im_end|>" in answer_text:
                answer_text = answer_text.split("<|im_end|>")[0].strip()
                if "Answer:" in answer_text:
                    answer_text = answer_text.split("Answer:")[1].strip()
                if "answer:" in answer_text:
                    answer_text = answer_text.split("answer:")[1].strip()
            if answer_text not in ["true", "false", "undetermined"]:
                if "true" in answer_text:
                    answer_text = "true"
                elif "false" in answer_text:
                    answer_text = "false"
                elif "undetermined" in answer_text:
                    answer_text = "undetermined"
                else:
                    answer_text = "None"

            prompts_dict.setdefault(current_prompt, []).append(answer_text)

    return prompts_dict


def extract_answers_simple(file_name: str) -> dict:

    with open(file_name, 'r') as file:
        content = file.read()

    prompts_dict = {}
    current_prompt = None
    iter = 0

    for i,line in enumerate(content.split('\n')):
        line = line.strip()
        
        if line.startswith("Prompt"):
            current_prompt = line
            iter += 1

        elif "TRUE" in line or "FALSE" in line or "UNDETERMINED" in line:

            for j in range(i-1,0,-1):
                if content.split('\n')[j] != "" and "Explanation" not in content.split('\n')[j]:
                    answer_text = content.split('\n')[j]
                    break

            if "<|im_end|>" in answer_text:
                answer_text = answer_text.split("<|im_end|>")[0].strip()
                if "Answer:" in answer_text:
                    answer_text = answer_text.split("Answer:")[1].strip()
                if "answer:" in answer_text:
                    answer_text = answer_text.split("answer:")[1].strip()
            
            answer_text = answer_text.lower().replace(".","").replace('\\','')
            if answer_text not in ["true", "false", "undetermined"]:
                if "true" in answer_text:
                    answer_text = "true"
                elif "false" in answer_text:
                    answer_text = "false"
                elif "undetermined" in answer_text:
                    answer_text = "undetermined"
                else:
                    answer_text = "None"

            prompts_dict.setdefault(current_prompt, []).append(answer_text)

    return prompts_dict



def evaluate_nli(y_pred:str, label:str) -> int:
    stemmer = PorterStemmer()
    y_pred = y_pred.rstrip('.,?!')
    label = label.rstrip('.,?!')
    stemmed_y_pred = stemmer.stem(y_pred)
    stemmed_label = stemmer.stem(label)

    return 1 if stemmed_label in stemmed_y_pred else 0


def compute_acc(imp_filename,
                ind_filename,
                inter_active_filename,
                passive_filename,
                prompt_format):

    if prompt_format == 'qa':

        imp_prompts_dict = extract_answers_qa(imp_filename)
        ind_prompts_dict = extract_answers_qa(ind_filename)
        inter_active_prompts_dict = extract_answers_qa(inter_active_filename)
        passive_prompts_dict = extract_answers_qa(passive_filename)
    
    elif prompt_format == 'simple':

        imp_prompts_dict = extract_answers_simple(imp_filename)
        ind_prompts_dict = extract_answers_simple(ind_filename)
        inter_active_prompts_dict = extract_answers_simple(inter_active_filename)
        passive_prompts_dict = extract_answers_simple(passive_filename)

    testset = EmbaseNLI().data

    imp_acc0 = 0
    imp_acc1 = 0
    imp_acc2 = 0

    ind_acc0 = 0
    ind_acc1 = 0
    ind_acc2 = 0

    inter_active_acc0 = 0
    inter_active_acc1 = 0
    inter_active_acc2 = 0

    passive_acc0 = 0
    passive_acc1 = 0
    passive_acc2 = 0

    for example in testset:
        label = example['output']

        imp_acc0 += evaluate_nli(imp_prompts_dict['Prompt 0'][testset.index(example)], label)
        imp_acc1 += evaluate_nli(imp_prompts_dict['Prompt 1'][testset.index(example)], label)
        imp_acc2 += evaluate_nli(imp_prompts_dict['Prompt 2'][testset.index(example)], label)

        ind_acc0 += evaluate_nli(ind_prompts_dict['Prompt 0'][testset.index(example)], label)
        ind_acc1 += evaluate_nli(ind_prompts_dict['Prompt 1'][testset.index(example)], label)
        ind_acc2 += evaluate_nli(ind_prompts_dict['Prompt 2'][testset.index(example)], label)

        inter_active_acc0 += evaluate_nli(inter_active_prompts_dict['Prompt 0'][testset.index(example)], label)
        inter_active_acc1 += evaluate_nli(inter_active_prompts_dict['Prompt 1'][testset.index(example)], label)
        inter_active_acc2 += evaluate_nli(inter_active_prompts_dict['Prompt 2'][testset.index(example)], label)

        passive_acc0 += evaluate_nli(passive_prompts_dict['Prompt 0'][testset.index(example)], label)
        passive_acc1 += evaluate_nli(passive_prompts_dict['Prompt 1'][testset.index(example)], label)
        passive_acc2 += evaluate_nli(passive_prompts_dict['Prompt 2'][testset.index(example)], label)

    imp_acc0 /= len(testset)
    imp_acc1 /= len(testset)
    imp_acc2 /= len(testset)

    ind_acc0 /= len(testset)
    ind_acc1 /= len(testset)
    ind_acc2 /= len(testset)

    inter_active_acc0 /= len(testset)
    inter_active_acc1 /= len(testset)
    inter_active_acc2 /= len(testset)

    passive_acc0 /= len(testset)
    passive_acc1 /= len(testset)
    passive_acc2 /= len(testset)

    imp_tab = PrettyTable(["Prompt", "Imperative - Accuracy"])
    input0 = ["Prompt 1"]
    input1 = ["Prompt 2"]
    input2 = ["Prompt 3"]

    imp_tab.add_row(input0 + [round(imp_acc0,4)])
    imp_tab.add_row(input1 + [round(imp_acc1,4)])
    imp_tab.add_row(input2 + [round(imp_acc2,4)])

    ind_tab = PrettyTable(["Prompt", "Indicative - Accuracy"])
    ind_tab.add_row(input0 + [round(ind_acc0,4)])
    ind_tab.add_row(input1 + [round(ind_acc1,4)])
    ind_tab.add_row(input2 + [round(ind_acc2,4)])

    inter_active_tab = PrettyTable(["Prompt", "Interrogative/Active - Accuracy"])
    inter_active_tab.add_row(input0 + [round(inter_active_acc0,4)])
    inter_active_tab.add_row(input1 + [round(inter_active_acc1,4)])
    inter_active_tab.add_row(input2 + [round(inter_active_acc2,4)])

    passive_tab = PrettyTable(["Prompt", "Passive - Accuracy"])
    passive_tab.add_row(input0 + [round(passive_acc0,4)])
    passive_tab.add_row(input1 + [round(passive_acc1,4)])
    passive_tab.add_row(input2 + [round(passive_acc2,4)])

    print(imp_tab)
    print(ind_tab)
    print(inter_active_tab)
    print(passive_tab)


if '__main__' == __name__:  

    prompt_formats = ['qa','simple']
    modes = ['0shot', '1shot', '3shot']

    for mode in modes:
        for prompt_format in prompt_formats:
                imp_filename = f'./openhermes/openhermes_outputs/NLI_imp/{prompt_format}/Embase_NLI_imp_{mode}_{prompt_format}.txt'
                ind_filename = f'./openhermes/openhermes_outputs/NLI_ind/{prompt_format}/Embase_NLI_ind_{mode}_{prompt_format}.txt'
                inter_active_filename = f'./openhermes/openhermes_outputs/NLI_inter_active/{prompt_format}/Embase_NLI_inter_active_{mode}_{prompt_format}.txt'
                passive_filename = f'./openhermes/openhermes_outputs/NLI_passive/{prompt_format}/Embase_NLI_passive_{mode}_{prompt_format}.txt'

                print(f"Mode: {mode}, Prompt Format: {prompt_format}")
                compute_acc(imp_filename, ind_filename, inter_active_filename, passive_filename, prompt_format)
                print("\n") 
