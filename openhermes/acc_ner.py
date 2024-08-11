import torch
import json
import random
import ast 
from prettytable import PrettyTable
import re 

def modify_text_within_brackets(text):
    
    new_text = ''
    within_brackets = False
    for i in range(len(text)):
        if text[i] == '[':
            within_brackets = True
        elif text[i] == ']':
            within_brackets = False
        elif within_brackets:
            if text[i:i+2] == ', ':
                if i > 0 and text[i - 1] != "'":
                    new_text += "'"
                if i < len(text) - 2 and text[i + 2] != "'":
                    new_text += "', "
                else:
                    new_text += ', '
                continue
        new_text += text[i]
    return new_text


def fix_string(input_string, prompt_format, mode):

    # replace multiple spaces with single space
    input_string = re.sub(' +', ' ', input_string)

    # check whether after [ follows a '. If not, add it.
    if "[" in input_string:
        if input_string[input_string.find("[")+1] != "'" and input_string[input_string.find("[")+1] != "]" and input_string[input_string.find("[")+1] != " " and input_string[input_string.find("[")+1] != '"':
            input_string = input_string[:input_string.find("[")+1] + "'" + input_string[input_string.find("[")+1:]
    # check before ] whether there is a '. If not, add it.
    if "]" in input_string:
        if input_string[input_string.find("]")-1] != "'" and input_string[input_string.find("]")-1] != "[" and input_string[input_string.find("]")-1] != '"':
            input_string = input_string[:input_string.find("]")] + "'" + input_string[input_string.find("]"):]

    input_string = modify_text_within_brackets(input_string)
    
    # check if after "{" there is a ' if not add it
    if "{" in input_string:
        if prompt_format == 'simple':
            if input_string[input_string.find("{")+1] != "'" and input_string[input_string.find("{")+1] != "}" and input_string[input_string.find("{")+1] != '"' and input_string[input_string.find("{")+2] != "'":
                input_string = input_string[:input_string.find("{")+1] + "'" + input_string[input_string.find("{")+1:]
        else:
            if mode != '3shot':
                if input_string[input_string.find("{")+1] != "'" and input_string[input_string.find("{")+1] != "}" and input_string[input_string.find("{")+1] != '"' and input_string[input_string.find("{")+2] != "'" and input_string[input_string.find("{")+3] != "'" and input_string[input_string.find("{")+4] != "'" and input_string[input_string.find("{")+5] != "'":
                    input_string = input_string[:input_string.find("{")+1] + "'" + input_string[input_string.find("{")+1:]


    # check if before : there is a ' if not add it
    if ":" in input_string:
        if input_string[input_string.find(":")-1] != "'" and input_string[input_string.find(":")-1] != " " and input_string[input_string.find(":")-1] != '"':
            input_string = input_string[:input_string.find(":")] + "'" + input_string[input_string.find(":"):]

    # after ] replace everything with }
    if "]" in input_string:
        input_string = input_string[:input_string.find("]")+1] + "}"


    return input_string



def extract_answers(file_name: str, prompt_format:str) -> dict:

    if prompt_format == "qa":

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
                    answer_text = content.split('\n')[i+1].lower()
                else:
                    answer_text = content.split('\n')[i+2].lower()
                    if answer_text == "{":
                        answer_text = answer_text + content.split('\n')[i+3]
                        if "}" not in answer_text:
                            answer_text = answer_text + content.split('\n')[i+4]

                if "<|im_end|>" in answer_text:
                    answer_text = answer_text.split("<|im_end|>")[0].strip()
                    if "Answer:" in answer_text:
                        answer_text = answer_text.split("Answer:")[1].strip()
                    if "answer:" in answer_text:
                        answer_text = answer_text.split("answer:")[1].strip()


                prompts_dict.setdefault(current_prompt, []).append(answer_text)

    elif prompt_format == "simple":

        with open(file_name, 'r') as file:
            content = file.read()

        prompts_dict = {}
        current_prompt = None
        
        for i,line in enumerate(content.split('\n')):
            line = line.strip()
            
            if line.startswith("Prompt"):
                current_prompt = line
                
            elif line == "-":
                answer_text = content.split('\n')[i-3].lower()

                if "<|im_end|>" in answer_text:
                    answer_text = answer_text.split("<|im_end|>")[0].strip()
                if "answer:" in answer_text:
                    answer_text = answer_text.split("answer:")[1].strip()

                prompts_dict.setdefault(current_prompt, []).append(answer_text)

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

            # True Positives: Intersection of predicted and actual values for the key
            tp += len(set(actual[key]) & set(predicted[key]))

            # False Positives: Values in predicted but not in actual for the key
            fp += len(set(predicted[key]) - set(actual[key]))

            # False Negatives: Values in actual but not in predicted for the key
            fn += len(set(actual[key]) - set(predicted[key]))
        else:
            # If the key is not in predicted, all actual values contribute to False Negatives
            fn += len(actual[key])

    return tp, fp, fn


def em_and_f1(prompts_dict: dict, testset:list, prompt_format:str, mode:str) -> None:
    
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

    count_invalid = 0
    fixed_words = ['none', 'not', 'any', 'missing', 'no']
    length = len(testset)


    for i,example in enumerate(testset):
        
        output = example['structured_output']
        output = str(output).replace('"', "'").replace(')','')     

        sample = prompts_dict['Prompt 0'][testset.index(example)]
        sample1 = prompts_dict['Prompt 1'][testset.index(example)]
        sample2 = prompts_dict['Prompt 2'][testset.index(example)]

        if any(word in sample for word in fixed_words) or (sample == "{}.") or ("device" in sample and "disease" in sample and "drug" in sample and "[]" in sample) or ("[]" in sample):
            sample = "{}"
        if any(word in sample1 for word in fixed_words) or (sample1 == "{}.") or ("device" in sample1 and "disease" in sample1 and "drug" in sample1 and "[]" in sample1) or ("[]" in sample1):
            sample1 = "{}"
        if any(word in sample2 for word in fixed_words) or (sample2 == "{}.") or ("device" in sample2 and "disease" in sample2 and "drug" in sample2 and "[]" in sample2) or ("[]" in sample2):
            sample2 = "{}"

        sample = sample[sample.find("{"):sample.find("}")+1]
        sample1 = sample1[sample1.find("{"):sample1.find("}")+1]
        sample2 = sample2[sample2.find("{"):sample2.find("}")+1]

        if (all(character not in sample for character in ["{","}"])) or (all(character not in sample1 for character in ["{","}"])) or (all(character not in sample2 for character in ["{","}"])) or (sample.count("[") > 1) or (sample1.count("[") > 1) or (sample2.count("[") > 1) or ("..." in sample) or ("..." in sample1) or ("..." in sample2)or (sample.count("{") > 1) or (sample1.count("{") > 1) or (sample2.count("{") > 1):
            print(f"Example {i} is invalid. Skipping...")
            length -= 1
            count_invalid += 1
            continue

        # Fix the string
        sample = fix_string(sample,prompt_format,mode).replace('"', "'").replace("\\", "").replace("''", "'").replace("[']","[]").replace("{{","{").replace("',  ","', '").replace("''","'").replace(", ']","]").replace("[ ']","[]")
        sample1 = fix_string(sample1,prompt_format,mode).replace('"', "'").replace("\\", "").replace("''", "'").replace("[']","[]").replace("'  '","'").replace(",']}","]}").replace(",  ']","]").replace("''","'").replace("{' '","{'").replace(",  ',  '",", '").replace("' '","'").replace("[']","[]").replace("[ ']","[]").replace("{'}","{}")
        sample2 = fix_string(sample2,prompt_format,mode).replace('"', "'").replace("\\", "").replace("''", "'").replace("[']","[]").replace("alzheimer's", "alzheimers").replace("'  '","'").replace(",',",",").replace("[ ']","[]").replace("''","'").replace("{'}","{}").replace("{' }","{}").replace("{' '","{'}").replace("' '","'")

        if "{" in sample and "}" not in sample:
            sample = sample + "}"
        if "}" in sample and "{" not in sample:
            sample = "{" + sample
        if "{" in sample1 and "}" not in sample1:
            sample1 = sample1 + "}"
        if "}" in sample1 and "{" not in sample1:
            sample1 = "{" + sample1
        if "{" in sample2 and "}" not in sample2:
            sample2 = sample2 + "}"
        if "}" in sample2 and "{" not in sample2:
            sample2 = "{" + sample2

        # Exact match
        if output.lower() == sample.lower():
            acc0 += 1
        if output.lower() == sample1.lower():
            acc1 += 1
        if output.lower() == sample2.lower():
            acc2 += 1

        output = output.lower()
        sample = sample.lower()
        sample1 = sample1.lower()
        sample2 = sample2.lower()


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

    print("Number of invalid examples: ", count_invalid)

    results = {
        "Exact Match Prompt 0": acc0 / length,
        "Precision Prompt 0": tpp_0 / (tpp_0 + fpp_0),
        "Recall Prompt 0": tpp_0 / (tpp_0 + fnn_0),
        "F1 Score Prompt 0": 2 * (tpp_0 / (tpp_0 + fpp_0)) * (tpp_0 / (tpp_0 + fnn_0)) / ((tpp_0 / (tpp_0 + fpp_0)) + (tpp_0 / (tpp_0 + fnn_0))),
        "Exact Match Prompt 1": acc1 / length,
        "Precision Prompt 1": tpp_1 / (tpp_1 + fpp_1),
        "Recall Prompt 1": tpp_1 / (tpp_1 + fnn_1),
        "F1 Score Prompt 1": 2 * (tpp_1 / (tpp_1 + fpp_1)) * (tpp_1 / (tpp_1 + fnn_1)) / ((tpp_1 / (tpp_1 + fpp_1)) + (tpp_1 / (tpp_1 + fnn_1))),
        "Exact Match Prompt 2": acc2 / length,
        "Precision Prompt 2": tpp_2 / (tpp_2 + fpp_2),
        "Recall Prompt 2": tpp_2 / (tpp_2 + fnn_2),
        "F1 Score Prompt 2": 2 * (tpp_2 / (tpp_2 + fpp_2)) * (tpp_2 / (tpp_2 + fnn_2)) / ((tpp_2 / (tpp_2 + fpp_2)) + (tpp_2 / (tpp_2 + fnn_2))),
    }

    tab = PrettyTable(["Prompt", "EM", "F1", "Precision", "Recall"])
    input0 = ["Prompt 1"]
    input1 = ["Prompt 2"]
    input2 = ["Prompt 3"]

    # Round to 4 decimal places
    tab.add_row(input0 + [exact_match_0, f1_0, precision_0, recall_0])
    tab.add_row(input1 + [exact_match_1, f1_1, precision_1, recall_1])
    tab.add_row(input2 + [exact_match_2, f1_2, precision_2, recall_2])
    print(tab)

    return results

if "__main__" == __name__:

    random.seed(0)
    testset = EmbaseNER().data
    random.shuffle(testset)
    testset = testset[:200]

    prompt_formats = ["simple","qa"]
    modes = ["0shot", "1shot", "3shot"]


    for prompt_format in prompt_formats:
        for mode in modes:
            print(f"Prompt Format: {prompt_format}, Mode: {mode}")
            print("\n")
            print("Imperative")
            imp_prompts_dict = extract_answers(f'./openhermes/openhermes_outputs/NER_imp/{prompt_format}/Embase_NER_imp_{mode}_{prompt_format}_True.txt', prompt_format)
            imp_results = em_and_f1(imp_prompts_dict, testset, prompt_format, mode)
            print("Indicative")
            ind_prompts_dict = extract_answers(f'./openhermes/openhermes_outputs/NER_ind/{prompt_format}/Embase_NER_ind_{mode}_{prompt_format}_True.txt', prompt_format)
            ind_results = em_and_f1(ind_prompts_dict, testset, prompt_format, mode)
            print("Interrogative/Active")
            inter_active_prompts_dict = extract_answers(f'./openhermes/openhermes_outputs/NER_inter_active/{prompt_format}/Embase_NER_inter_active_{mode}_{prompt_format}_True.txt', prompt_format)
            inter_active_results = em_and_f1(inter_active_prompts_dict, testset, prompt_format, mode)
            print("Passive")
            passive_prompts_dict = extract_answers(f'./openhermes/openhermes_outputs/NER_passive/{prompt_format}/Embase_NER_passive_{mode}_{prompt_format}_True.txt', prompt_format)
            passive_results = em_and_f1(passive_prompts_dict, testset, prompt_format, mode)
            print("\n")

            