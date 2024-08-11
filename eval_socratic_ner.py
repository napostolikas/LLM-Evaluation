import re
from prettytable import PrettyTable

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
    

def extract_metrics(prompts_dict):

    tp_pattern = r'(?:TP|\(TP\)|True\s+Positives).*?(\d+)'
    fp_pattern = r'(?:FP|\(FP\)|False\s+Positives).*?(\d+)'
    fn_pattern = r'(?:FN|\(FN\)|False\s+Negatives).*?(\d+)'

    metrics = {}

    for prompt, conv_list in prompts_dict.items():

        tp = 0
        fp = 0
        fn = 0
        em = 0 

        for conv in conv_list:
            
            tp_matches = re.findall(tp_pattern, conv)
            fp_matches = re.findall(fp_pattern, conv)
            fn_matches = re.findall(fn_pattern, conv)

            if not (tp_matches or fp_matches or fn_matches):
                print("Conversation skipped")
                continue

            tp_numbers = [int(match) for match in tp_matches][-1] if tp_matches else 0
            fp_numbers = [int(match) for match in fp_matches][-1] if fp_matches else 0
            fn_numbers = [int(match) for match in fn_matches][-1] if fn_matches else 0

            tp += tp_numbers
            fp += fp_numbers
            fn += fn_numbers

            if tp_numbers > 0 and fp_numbers == 0 and fn_numbers == 0:
                em += 1

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        em = em / len(conv_list)

        metrics[prompt] = {'Precision': round(precision,4), 'Recall': round(recall,4), 'F1': round(f1,4), 'EM': round(em,4)}

    tab = PrettyTable(["Prompt", "EM", "F1", "Precision", "Recall"])
    input0 = ["Prompt 0"]
    input1 = ["Prompt 1"]
    input2 = ["Prompt 2"]

    tab.add_row(input0 + [metrics['Prompt 0']['EM'], metrics['Prompt 0']['F1'], metrics['Prompt 0']['Precision'], metrics['Prompt 0']['Recall']])
    tab.add_row(input1 + [metrics['Prompt 1']['EM'], metrics['Prompt 1']['F1'], metrics['Prompt 1']['Precision'], metrics['Prompt 1']['Recall']])
    tab.add_row(input2 + [metrics['Prompt 2']['EM'], metrics['Prompt 2']['F1'], metrics['Prompt 2']['Precision'], metrics['Prompt 2']['Recall']])

    return metrics, tab


if __name__ == "__main__":

    model_name = 'starling'

    ind_prompts_dict, imp_prompts_dict, inter_active_prompts_dict, passive_prompts_dict = extract_conv(f"./{model_name}/{model_name}_outputs/{model_name}_socratic_NER.txt")

    ind_metrics, ind_tab = extract_metrics(ind_prompts_dict)
    imp_metrics, imp_tab = extract_metrics(imp_prompts_dict)
    inter_metrics, inter_tab = extract_metrics(inter_active_prompts_dict)
    passive_metrics, passive_tab = extract_metrics(passive_prompts_dict)

    print("Ind Metrics")
    print(ind_tab)
    print("Imp Metrics")
    print(imp_tab)
    print("Inter_Active Metrics")
    print(inter_tab)
    print("Passive Metrics")
    print(passive_tab)

