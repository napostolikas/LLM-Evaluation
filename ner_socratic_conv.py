import transformers
import re
import torch
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

class SocraticGPT:
    def __init__(self, role, model, tokenizer):
        self.role = role
        self.tokenizer = tokenizer
        self.model = model

    def get_response(self, input_prompt):
        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to('cuda')
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=300,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response_ids = outputs[0]
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response_text


def dialogue(model, tokenizer, ground_truth, candidate_answer):

    socrates = SocraticGPT(role="Socrates", model=model, tokenizer=tokenizer)
    theaetetus = SocraticGPT(role="Theaetetus", model=model, tokenizer=tokenizer)    
    
    socrates_input_prompt = f'''GPT4 Correct System: From now on you are Socrates and you will chat with Theaetetus. You will be engaged in a dialogue and you will determine the true positives, the false positives and the false negatives between a candidate answer and the ground truth.\nTrue Positives: the entities that appear in both the candidate answer and the ground truth.\n False Positives: the entities that appear in the candidate answer but not in the ground truth.\n False Negatives: the entities that appear in the ground truth but not in the candidate answer.\n The ground truth is: \"[{ground_truth}]\". The candidate answer is: \"[{candidate_answer}]\".\nOutput format: FP = [number], TP = [number], FN = [number]<|end_of_turn|>GPT4 Correct User: Start the conversation by saying "Hi Theaetetus, let's work together to find the True Positives, False Positives and False Negatives between the ground truth and a candidate answer". Then provide your answer and ask for Theaetetus' opinion.<|end_of_turn|>GPT4 Correct Assistant:<|end_of_turn|>'''
    socrates_response = socrates.get_response(socrates_input_prompt)
    socrates_short_response = socrates_response.split('GPT4 Correct Assistant:')[1].replace("\n","")
    print("Socrates:", socrates_short_response)
    print("\n")
    
    theaetetus_input_prompt = f"GPT4 Correct System: From now on you are Theatetus and you will chat with Socrates. You will be engaged in a dialogue and you will construct a dictionary output of the entities in the text given. The text given is a model's output for a named entity recognition task.\nThe entities you should look for are 'device', 'disease', 'drug' or 'none'. The text is: \"[{candidate_answer}]\".\nOutput format: {{'entity_type': ['entity1', 'entity2', ... 'entityN']}}\nMake sure to say 'Hi Socrates' and be friendly.<|end_of_turn|>GPT4 Correct User: {socrates_short_response}<|end_of_turn|>GPT4 Correct Assistant:<|end_of_turn|>"
    theaetetus_response = theaetetus.get_response(theaetetus_input_prompt)
    theaetetus_short_response = theaetetus_response.split('GPT4 Correct Assistant:')[1].replace("\n","")
    print("Theaetetus:", theaetetus_short_response)
    print("\n")
  
    conversation = [socrates_short_response, theaetetus_short_response]
    return conversation
    

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

    
if __name__ == '__main__':

    model_name = 'starling'

    ind_prompts_dict = extract_answers(f'./{model_name}/{model_name}_outputs/NER_ind/qa_true/Embase_NER_ind_0shot_qa_true.txt')
    imp_prompts_dict = extract_answers(f'./{model_name}/{model_name}_outputs/NER_imp/qa_true/Embase_NER_imp_0shot_qa_true.txt')
    inter_active_prompts_dict = extract_answers(f'./{model_name}/{model_name}_outputs/NER_inter_active/qa_true/Embase_NER_inter_active_0shot_qa_true.txt')
    passive_prompts_dict = extract_answers(f'./{model_name}/{model_name}_outputs/NER_passive/qa_true/Embase_NER_passive_0shot_qa_true.txt')

    testset = EmbaseNER().data

    model = transformers.AutoModelForCausalLM.from_pretrained("berkeley-nest/Starling-LM-7B-alpha", torch_dtype = torch.float16)
    tokenizer = transformers.AutoTokenizer.from_pretrained("berkeley-nest/Starling-LM-7B-alpha")
    model = model.to('cuda')

    for i,example in enumerate(testset):
    
        question = example['question']
        context = example['context']
        answer = example['output']

        ind_candidate_answer0 = ind_prompts_dict['Prompt 0'][testset.index(example)]
        ind_candidate_answer1 = ind_prompts_dict['Prompt 1'][testset.index(example)]
        ind_candidate_answer2 = ind_prompts_dict['Prompt 2'][testset.index(example)]

        imp_candidate_answer0 = imp_prompts_dict['Prompt 0'][testset.index(example)]
        imp_candidate_answer1 = imp_prompts_dict['Prompt 1'][testset.index(example)]
        imp_candidate_answer2 = imp_prompts_dict['Prompt 2'][testset.index(example)]

        inter_active_candidate_answer0 = inter_active_prompts_dict['Prompt 0'][testset.index(example)]
        inter_active_candidate_answer1 = inter_active_prompts_dict['Prompt 1'][testset.index(example)]
        inter_active_candidate_answer2 = inter_active_prompts_dict['Prompt 2'][testset.index(example)]

        passive_candidate_answer0 = passive_prompts_dict['Prompt 0'][testset.index(example)]
        passive_candidate_answer1 = passive_prompts_dict['Prompt 1'][testset.index(example)]
        passive_candidate_answer2 = passive_prompts_dict['Prompt 2'][testset.index(example)]

        ind_score0 = dialogue(model, tokenizer, question, context, answer, ind_candidate_answer0)
        ind_score1 = dialogue(model, tokenizer, question, context, answer, ind_candidate_answer1)
        ind_score2 = dialogue(model, tokenizer, question, context, answer, ind_candidate_answer2)

        imp_score0 = dialogue(model, tokenizer, question, context, answer, imp_candidate_answer0)
        imp_score1 = dialogue(model, tokenizer, question, context, answer, imp_candidate_answer1)
        imp_score2 = dialogue(model, tokenizer, question, context, answer, imp_candidate_answer2)

        inter_active_score0 = dialogue(model, tokenizer, question, context, answer, inter_active_candidate_answer0)
        inter_active_score1 = dialogue(model, tokenizer, question, context, answer, inter_active_candidate_answer1)
        inter_active_score2 = dialogue(model, tokenizer, question, context, answer, inter_active_candidate_answer2)

        passive_score0 = dialogue(model, tokenizer, question, context, answer, passive_candidate_answer0)
        passive_score1 = dialogue(model, tokenizer, question, context, answer, passive_candidate_answer1)
        passive_score2 = dialogue(model, tokenizer, question, context, answer, passive_candidate_answer2)

        # write scores to file

        with open('scores.txt', 'a') as file:
            file.write(f'Example {testset.index(example)}\n')
            file.write(f'Ind Score 0: {ind_score0}\n')
            file.write(f'Ind Score 1: {ind_score1}\n')
            file.write(f'Ind Score 2: {ind_score2}\n')
            file.write(f'Imp Score 0: {imp_score0}\n')
            file.write(f'Imp Score 1: {imp_score1}\n')
            file.write(f'Imp Score 2: {imp_score2}\n')
            file.write(f'Inter Active Score 0: {inter_active_score0}\n')
            file.write(f'Inter Active Score 1: {inter_active_score1}\n')
            file.write(f'Inter Active Score 2: {inter_active_score2}\n')
            file.write(f'Passive Score 0: {passive_score0}\n')
            file.write(f'Passive Score 1: {passive_score1}\n')
            file.write(f'Passive Score 2: {passive_score2}\n')
            file.write('---')

