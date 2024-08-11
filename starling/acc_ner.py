import torch
import json
import random
import ast 
from prettytable import PrettyTable
import csv

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


def calculate_confusion_matrix(actual, predicted):

    tp = 0
    fp = 0
    fn = 0

    for key in actual:
        if key in predicted:
            
            predicted[key] = [str(word).replace("(","").replace(")","").capitalize() for word in predicted[key]]
            actual[key] = [word.capitalize() for word in actual[key]]
            tp += len(set(actual[key]) & set(predicted[key]))
            fp += len(set(predicted[key]) - set(actual[key]))
            fn += len(set(actual[key]) - set(predicted[key]))
        else:
            fn += len(actual[key])

    return tp, fp, fn


def em_and_f1(prompts_dict: dict, testset:list) -> None:
    
    acc0 = 0 
    acc1 = 0
    acc2 = 0

    tpp_0 = 0
    fpp_0 = 0
    fnn_0 = 0

    tpp_1 = 0
    fpp_1 = 0
    fnn_1 = 0

    tpp_2 = 0
    fpp_2 = 0
    fnn_2 = 0

    length = len(testset)
    count_invalid = 0
    fixed_words = ['none', 'not', 'any', 'missing', 'no']

    for i, example in enumerate(testset):

        output = example['structured_output']

        output = str(output).replace('"', "'").replace(')','')     

        sample = prompts_dict['Prompt 0'][testset.index(example)]
        sample1 = prompts_dict['Prompt 1'][testset.index(example)]
        sample2 = prompts_dict['Prompt 2'][testset.index(example)]
        
        # if ("{" in sample and "}" not in sample) or ("{" not in sample):
        #     sample = "{}"
        # if "{" in sample1 and "}" not in sample1 or ("{" not in sample1):
        #     sample1 = "{}"
        # if "{" in sample2 and "}" not in sample2 or ("{" not in sample2):
        #     sample2 = "{}"

        if any(word in sample for word in fixed_words) or (sample == "{}.") or ("device" in sample and "disease" in sample and "drug" in sample and "[]" in sample) or ("[]" in sample):
            sample = "{}"
        if any(word in sample1 for word in fixed_words) or (sample1 == "{}.") or ("device" in sample1 and "disease" in sample1 and "drug" in sample1 and "[]" in sample1) or ("[]" in sample1):
            sample1 = "{}"
        if any(word in sample2 for word in fixed_words) or (sample2 == "{}.") or ("device" in sample2 and "disease" in sample2 and "drug" in sample2 and "[]" in sample2) or ("[]" in sample2):
            sample2 = "{}"

        sample = sample[sample.find("{"):sample.find("}")+1]
        sample1 = sample1[sample1.find("{"):sample1.find("}")+1]
        sample2 = sample2[sample2.find("{"):sample2.find("}")+1]
            
        if (all(character not in sample for character in ["{","}"])) or (all(character not in sample1 for character in ["{","}"])) or (all(character not in sample2 for character in ["{","}"])) or (sample.count("[") > 1) or (sample1.count("[") > 1) or (sample2.count("[") > 1):
            print(f"Example {i} is invalid. Skipping...")
            length -= 1
            count_invalid += 1
            continue


        # Exact match
        if output.lower() == sample.replace('"', "'").lower():
            acc0 += 1
        if output.lower() == sample1.replace('"', "'").lower():
            acc1 += 1
        if output.lower() == sample2.replace('"', "'").lower():
            acc2 += 1


        if (sample != "{}") and (output != "{}"):
            tp_0, fp_0, fn_0 = calculate_confusion_matrix(ast.literal_eval(output), ast.literal_eval(sample))
            tpp_0 += tp_0
            fpp_0 += fp_0
            fnn_0 += fn_0
        elif (sample == "{}") and (output != "{}"):
            tp_0, fp_0, fn_0 = 0, 0, len(ast.literal_eval(output))
            tpp_0 += tp_0
            fpp_0 += fp_0
            fnn_0 += fn_0
        elif (sample != "{}") and (output == "{}"):
            tp_0, fp_0, fn_0 = 0, len(ast.literal_eval(sample)), 0
            tpp_0 += tp_0
            fpp_0 += fp_0
            fnn_0 += fn_0


        if (sample1 != "{}") and (output != "{}"):
            tp_1, fp_1, fn_1 = calculate_confusion_matrix(ast.literal_eval(output), ast.literal_eval(sample1))
            tpp_1 += tp_1
            fpp_1 += fp_1
            fnn_1 += fn_1

        elif (sample1 == "{}") and (output != "{}"):
            tp_1, fp_1, fn_1 = 0, 0, len(ast.literal_eval(output))
            tpp_1 += tp_1
            fpp_1 += fp_1
            fnn_1 += fn_1

        elif (sample1 != "{}") and (output == "{}"):
            tp_1, fp_1, fn_1 = 0, len(ast.literal_eval(sample1)), 0
            tpp_1 += tp_1
            fpp_1 += fp_1
            fnn_1 += fn_1

        if (sample2 != "{}") and (output != "{}"):
            tp_2, fp_2, fn_2 = calculate_confusion_matrix(ast.literal_eval(output), ast.literal_eval(sample2))
            tpp_2 += tp_2
            fpp_2 += fp_2
            fnn_2 += fn_2

        elif (sample2 == "{}") and (output != "{}"):
            tp_2, fp_2, fn_2 = 0, 0, len(ast.literal_eval(output))
            tpp_2 += tp_2
            fpp_2 += fp_2
            fnn_2 += fn_2

        elif (sample2 != "{}") and (output == "{}"):
            tp_2, fp_2, fn_2 = 0, len(ast.literal_eval(sample2)), 0
            tpp_2 += tp_2
            fpp_2 += fp_2
            fnn_2 += fn_2

    exact_match_0 = round(acc0 / length, 4)
    exact_match_1 = round(acc1 / length, 4)
    exact_match_2 = round(acc2 / length, 4)

    precision_0 = round(tpp_0 / (tpp_0 + fpp_0), 4) 
    precision_1 = round(tpp_1 / (tpp_1 + fpp_1), 4)
    precision_2 = round(tpp_2 / (tpp_2 + fpp_2), 4)

    recall_0 = round(tpp_0 / (tpp_0 + fnn_0), 4)
    recall_1 = round(tpp_1 / (tpp_1 + fnn_1), 4)
    recall_2 = round(tpp_2 / (tpp_2 + fnn_2), 4)

    f1_0 = round(2 * (tpp_0 / (tpp_0 + fpp_0)) * (tpp_0 / (tpp_0 + fnn_0)) / ((tpp_0 / (tpp_0 + fpp_0)) + (tpp_0 / (tpp_0 + fnn_0))), 4)
    f1_1 = round(2 * (tpp_1 / (tpp_1 + fpp_1)) * (tpp_1 / (tpp_1 + fnn_1)) / ((tpp_1 / (tpp_1 + fpp_1)) + (tpp_1 / (tpp_1 + fnn_1))), 4)
    f1_2 = round(2 * (tpp_2 / (tpp_2 + fpp_2)) * (tpp_2 / (tpp_2 + fnn_2)) / ((tpp_2 / (tpp_2 + fpp_2)) + (tpp_2 / (tpp_2 + fnn_2))), 4)

    results = {
        "Exact Match Prompt 0": acc0 / len(testset),
        "Precision Prompt 0": tpp_0 / (tpp_0 + fpp_0),
        "Recall Prompt 0": tpp_0 / (tpp_0 + fnn_0),
        "F1 Score Prompt 0": 2 * (tpp_0 / (tpp_0 + fpp_0)) * (tpp_0 / (tpp_0 + fnn_0)) / ((tpp_0 / (tpp_0 + fpp_0)) + (tpp_0 / (tpp_0 + fnn_0))),
        "Exact Match Prompt 1": acc1 / len(testset),
        "Precision Prompt 1": tpp_1 / (tpp_1 + fpp_1),
        "Recall Prompt 1": tpp_1 / (tpp_1 + fnn_1),
        "F1 Score Prompt 1": 2 * (tpp_1 / (tpp_1 + fpp_1)) * (tpp_1 / (tpp_1 + fnn_1)) / ((tpp_1 / (tpp_1 + fpp_1)) + (tpp_1 / (tpp_1 + fnn_1))),
        "Exact Match Prompt 2": acc2 / len(testset),
        "Precision Prompt 2": tpp_2 / (tpp_2 + fpp_2),
        "Recall Prompt 2": tpp_2 / (tpp_2 + fnn_2),
        "F1 Score Prompt 2": 2 * (tpp_2 / (tpp_2 + fpp_2)) * (tpp_2 / (tpp_2 + fnn_2)) / ((tpp_2 / (tpp_2 + fpp_2)) + (tpp_2 / (tpp_2 + fnn_2))),
    }

    tab = PrettyTable(["Prompt", "EM", "F1", "Precision", "Recall"])
    input0 = ["Prompt 1"]
    input1 = ["Prompt 2"]
    input2 = ["Prompt 3"]

    tab.add_row(input0 + [exact_match_0, f1_0, precision_0, recall_0])
    tab.add_row(input1 + [exact_match_1, f1_1, precision_1, recall_1])
    tab.add_row(input2 + [exact_match_2, f1_2, precision_2, recall_2])
    print("Invalid Examples:", count_invalid)
    print(tab)

    return results, tab

if "__main__" == __name__:

    random.seed(0)
    testset = EmbaseNER().data
    random.shuffle(testset)
    testset = testset[:200]

    prompt_formats = ["qa", "simple"]
    modes = ["0shot", "1shot", "3shot"]


    for prompt_format in prompt_formats:
        for mode in modes:
            print(f"Prompt Format: {prompt_format}, Mode: {mode}")
            print("\n")
            print("Imperative")
            imp_prompts_dict = extract_answers(f'./starling/starling_outputs/NER_imp/{prompt_format}_true/Embase_NER_imp_{mode}_{prompt_format}_True.txt')
            imp_results, imp_tab = em_and_f1(imp_prompts_dict, testset)
            print("Indicative")
            ind_prompts_dict = extract_answers(f'./starling/starling_outputs/NER_ind/{prompt_format}_true/Embase_NER_ind_{mode}_{prompt_format}_True.txt')
            ind_results, ind_tab = em_and_f1(ind_prompts_dict, testset)
            print("Interrogative/Active")
            inter_active_prompts_dict = extract_answers(f'./starling/starling_outputs/NER_inter_active/{prompt_format}_true/Embase_NER_inter_active_{mode}_{prompt_format}_True.txt')
            inter_active_results, inter_active_tab = em_and_f1(inter_active_prompts_dict, testset)
            print("Passive")
            passive_prompts_dict = extract_answers(f'./starling/starling_outputs/NER_passive/{prompt_format}_true/Embase_NER_passive_{mode}_{prompt_format}_True.txt')
            passive_results, passive_tab = em_and_f1(passive_prompts_dict, testset)
            print("\n")
