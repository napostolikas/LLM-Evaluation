import re
from prettytable import PrettyTable
import torch
import json
import pandas as pd

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
    
def extract_answers(file_name: str) -> dict:

    with open(file_name, 'r') as file:
        content = file.read()

    prompts_dict = {}
    current_prompt = None
    current_value = None

    for line in content.split('\n'):
        line = line.strip()
        
        if line.startswith("Prompt"):
            current_prompt = line

        elif "GPT4 Correct Assistant" in line:            
            current_value = line.split("Assistant:")[1].strip()
            prompts_dict.setdefault(current_prompt, []).append(current_value)
    return prompts_dict

def extract_conv(file_name: str):

    with open(file_name, 'r') as file:
        content = file.read()

        ind_prompts_dict = {}
        imp_prompts_dict = {}
        inter_active_prompts_dict = {}
        passive_prompts_dict = {}

        for line in content.split('\n'):
            line = line.strip()
            
            if line.startswith("Ind"):
                if "Ind Score 0" in line:
                    current_prompt = "Prompt 0"
                    current_value = line.split("Ind Score 0:")[1].strip()
                elif "Ind Score 1" in line:
                    current_prompt = "Prompt 1"
                    current_value = line.split("Ind Score 1:")[1].strip()

                elif "Ind Score 2" in line:
                    current_prompt = "Prompt 2"
                    current_value = line.split("Ind Score 2:")[1].strip()
                
                ind_prompts_dict.setdefault(current_prompt, []).append(current_value)

            elif line.startswith("Imp"):
                if "Imp Score 0" in line:
                    current_prompt = "Prompt 0"
                    current_value = line.split("Imp Score 0:")[1].strip()

                elif "Imp Score 1" in line:
                    current_prompt = "Prompt 1"
                    current_value = line.split("Imp Score 1:")[1].strip()

                elif "Imp Score 2" in line:
                    current_prompt = "Prompt 2"
                    current_value = line.split("Imp Score 2:")[1].strip()

                imp_prompts_dict.setdefault(current_prompt, []).append(current_value)

            elif line.startswith("Inter Active Score"):
                if "Inter Active Score 0" in line:
                    current_prompt = "Prompt 0"
                    current_value = line.split("Inter Active Score 0:")[1].strip()

                elif "Inter Active Score 1" in line:
                    current_prompt = "Prompt 1"
                    current_value = line.split("Inter Active Score 1:")[1].strip()

                elif "Inter Active Score 2" in line:
                    current_prompt = "Prompt 2"
                    current_value = line.split("Inter Active Score 2:")[1].strip()

                elif "Inter Active Score 3" in line:
                    current_prompt = "Prompt 3"
                    current_value = line.split("Inter Active Score 3:")[1].strip()

                inter_active_prompts_dict.setdefault(current_prompt, []).append(current_value)
            
            elif line.startswith("Passive Score"):
                if "Passive Score 0" in line:
                    current_prompt = "Prompt 0"
                    current_value = line.split("Passive Score 0:")[1].strip()

                elif "Passive Score 1" in line:
                    current_prompt = "Prompt 1"
                    current_value = line.split("Passive Score 1:")[1].strip()

                elif "Passive Score 2" in line:
                    current_prompt = "Prompt 2"
                    current_value = line.split("Passive Score 2:")[1].strip()

                elif "Passive Score 3" in line:
                    current_prompt = "Prompt 3"
                    current_value = line.split("Passive Score 3:")[1].strip()

                passive_prompts_dict.setdefault(current_prompt, []).append(current_value)

        return ind_prompts_dict, imp_prompts_dict, inter_active_prompts_dict, passive_prompts_dict
    
def extract_scores(prompts_dict: dict) -> dict:
    
    scores_dict = {}

    for prompt, conv_list in prompts_dict.items():
        scores_list = []
        for i,conv in enumerate(conv_list):
            matches = re.findall(r'\[.*?\]', conv)
            socrates_opinion = None
            theaetetus_opinion = None
            count = 0

            for j in range(len(matches)):
                if matches[j] == "[factual]" or matches[j] == "[not factual]":
                    count += 1
                    if count == 1:
                        socrates_opinion = matches[j]
                    elif count == 2:
                        theaetetus_opinion = matches[j]
                        break

            if socrates_opinion == '[factual]' and theaetetus_opinion == '[factual]':
                scores_list.append(float(1))
            elif socrates_opinion == '[factual]' and theaetetus_opinion == '[not factual]':
                # print(f"S + | T - | Example {i}")
                scores_list.append(float(1))
            elif socrates_opinion == '[not factual]' and theaetetus_opinion == '[factual]':
                # print(f"S - | T + | Example {i}")
                scores_list.append(float(0))
            elif socrates_opinion == '[not factual]' and theaetetus_opinion == '[not factual]':
                scores_list.append(float(0))
            else:
                if '[factual]' in conv or 'factual' in conv:
                    scores_list.append(float(1))
                elif '[not factual]' in conv or 'not factual' in conv:
                    scores_list.append(float(0))
                    
        scores_dict[prompt] = scores_list

    return scores_dict

def calculate_accuracy(scores_dict: dict) -> dict:
    accuracy_dict = {}
    for prompt, scores_list in scores_dict.items():
        total_examples = len(scores_list)
        correct_predictions = sum(scores_list)
        accuracy = correct_predictions / total_examples
        accuracy_dict[prompt] = round(accuracy,4)
    return accuracy_dict

if __name__ == "__main__":

    model_name = 'starling'
    # model_name = 'openhermes'
    # model_name = 'openchat'

    file_name = f"./{model_name}_OQA_scores_factual.txt"

    ind_prompts_dict, imp_prompts_dict, inter_active_prompts_dict, passive_prompts_dict = extract_conv(file_name)

    ind_scores_dict = extract_scores(ind_prompts_dict)
    imp_scores_dict = extract_scores(imp_prompts_dict)
    inter_active_scores_dict = extract_scores(inter_active_prompts_dict)
    passive_scores_dict = extract_scores(passive_prompts_dict)

    ind_accuracy_dict = calculate_accuracy(ind_scores_dict)
    imp_accuracy_dict = calculate_accuracy(imp_scores_dict)
    inter_active_accuracy_dict = calculate_accuracy(inter_active_scores_dict)
    passive_accuracy_dict = calculate_accuracy(passive_scores_dict)

    table = PrettyTable()
    table.field_names = ["Prompt", "Indicative", "Imperative", "Interrogative", "Passive", "Active"]

    for prompt in ind_scores_dict.keys():
        table.add_row([prompt, ind_accuracy_dict[prompt], imp_accuracy_dict[prompt], inter_active_accuracy_dict[prompt], passive_accuracy_dict[prompt], inter_active_accuracy_dict[prompt]])
    table.add_row(["Prompt 3", " - ", " - ", " - ", passive_accuracy_dict["Prompt 3"], inter_active_accuracy_dict["Prompt 3"]])
    print(table)




