import os
import sys

# 获取当前文件所在的文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级文件夹路径
parent_dir = os.path.dirname(current_dir)
# 将上一级文件夹路径加入到sys.path中
sys.path.append(parent_dir)

from transformers import LlamaTokenizer, LlamaForCausalLM,GenerationConfig
import torch
import transformers 
from typing import List, Literal, Optional, Tuple, TypedDict
#from PreferCodeLlama.load_data import processClass
from load_data import processClass 
from arguments import train_config
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from llama_recipes.utils.memory_utils import MemoryTrace
PATH_TO_CONVERTED_WEIGHTS = "../CodeLlama-7b-Instruct-hf"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def evaluate(
            model,
            tokenizer,
            input=None,
            num_beams=1,
            max_new_tokens=2048,
            **kwargs,
    ):

        generation_config = GenerationConfig(
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
   
        #print(output.split(E_INST)[1])
        return output
    
def filte_code(text):
    text = text.split(E_INST)[1]
    text = text.strip()
    text = text.strip('\n')
    substring = "```"
    if text.find(substring) != -1: 
        text = text.split(substring)
        Len = len(text)
        if Len == 1:
            return text.strip('\n')
        else:
            if len(text[0]) > len(text[1]):
                return text[0].strip('\n')
            else:
                return text[1].strip('\n')
    else:
        return text 
    
def predict_eval_test_data():
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0]
    
    model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    model = model.cuda()
    tokenizer = LlamaTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    tokenizer.padding_side = "left"
    model.config.pad_token_id = 0
    tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    # load data
    #---------------------------------------------------------------------------------  
    proc = processClass()
    #train_data_set = proc.get_train_dataset(args,tokenizer)
    test_data_set = proc.get_test_datasets(args,tokenizer,is_test = True)    
    args.idnum = proc.idnum
    test_dataloader = DataLoader(test_data_set , batch_size= args.per_device_test_batch_size, collate_fn=test_data_set.collate_batch, num_workers=4)
    generation_json = []
    for step, batch in enumerate(tqdm(test_dataloader,colour="green", desc="predict Epoch", dynamic_ncols=True)):
        with MemoryTrace() as memtrace:
            user_id = batch['user_id']
            input_ids = batch['input_ids'].cuda() #test应该只有问题而没有回答
            code_lables = batch['code_labels'] 
            problem_id  = batch['problem_id']
            batch_output = evaluate(model, tokenizer,input_ids, pad_token_id = 0)
            for i in range(len(batch_output)):
                code_reply = filte_code(batch_output[i])
                item = {"user_id":str(user_id[i].item()),"problem_id":str(problem_id[i]),"code_lables":code_lables[i],"code_reply":code_reply}
                generation_json.append(item)
            
                        
    json.dump(
        generation_json,
        open("./out_predict/result_part_base100.json", 'w'),
        indent=4,
        ensure_ascii=False
    )

def get_instruction(input, language, is_test = False):
        instruction =B_SYS + f"Give you a Programming problem,please Provide answers in {language}"+ E_SYS + input
        if is_test:
            text = f"{B_INST} {(instruction).strip()} {E_INST}"
       
        return text 
 
 
def load_json_data(data_path):
    #print("loading text-score dataset from: \n   {}".format(data_path))
    data_list = []
    with open(data_path, 'r') as file:
        for line in file:
            # 逐行解析JSON对象
            data = json.loads(line)
            data_list.append(data)
    return data_list
       
def generate_humeval_data(data_path):
    data_list = load_json_data(data_path)
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0]
    
    model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    model = model.cuda()
    tokenizer = LlamaTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    tokenizer.padding_side = "left"
    model.config.pad_token_id = 0
    tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    # load data
    #---------------------------------------------------------------------------------  
    generation_json = []
    for item in data_list:
        input = item['prompt']
        input_ids = get_instruction(input,language="Java",is_test=True)
        task_id = item['task_id']
        batch_output = evaluate(model, tokenizer,input_ids, pad_token_id = 0)
        code_reply = filte_code(batch_output)
        item = {"task_id":task_id,"generation":code_reply}
        generation_json.append(item)
    
    with open("./out_predict/humeval_java_base.json", 'w') as file:
        for item in generation_json:
            json.dump(item, file)
            file.write('\n')
        
        
    
if __name__ == "__main__":
    #predict_eval_test_data()
    generate_humeval_data("./humaneval/humaneval_java.jsonl")
            
            