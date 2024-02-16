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
            max_new_tokens=4096,
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
    
def remove_java_prefix(input_string):
    # 检查字符串是否以"java"开头
    if input_string.startswith("java\n"):
        # 如果是，则删除开头的"java"
        result_string = input_string[len("java\n"):]
        return result_string
    else:
        # 如果不是，则返回原始字符串
        return input_string
    
def filte_code(text):
    text = text.split(E_INST)[1]
    text = text.strip()
    text = text.strip('\n')
    substring = "```"
    result_code = None
    if text.find(substring) != -1: 
        text = text.split(substring)
        Len = len(text)
        if Len == 1:
            result_code = text.strip('\n')
        else:
            if len(text[1]) == 0:
                result_code =  text[0].strip('\n')
            else:
                result_code =  text[1].strip('\n')
    else:
        result_code = text
    result_code  = remove_java_prefix(result_code)
    return result_code

def filte_code_for_humaneval(text):
    text = text.split(E_INST)[1]
    text = text.strip()
    text = text.strip('\n')
   
    #print(text)
    #input()
    #text = text.split(E_INST)[1]
    substring = "```"
    result_code = None
    if text.find(substring) != -1: 
        text = text.split(substring)
        Len = len(text)
        if Len == 1:
            result_code = text.strip('\n')
        else:
            if len(text[1]) == 0:
                result_code =  text[0].strip('\n')
            else:
                result_code =  text[1].strip('\n')
    else:
        result_code = text
   
    return result_code
    

def filte_origin_code(text):
    text = text.split(E_INST)[1]
    text = text.strip()
    text = text.strip('\n')
    return text
 
def predict_eval_test_data(out_predict__path, mode = "filt"):
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
    train_data_set = proc.get_train_dataset(args,tokenizer, is_test = False)
    eval_dataset_set = proc.get_eval_datasets(args,tokenizer,  is_test = False)
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
                if mode == "filt":
                    #print("here")
                    code_reply = filte_code(batch_output[i])
                    #input()
                    #print(code_reply)
                    #input()
                else :
                    code_reply = filte_origin_code(batch_output[i])
                item = {"user_id":str(user_id[i].item()),"problem_id":str(problem_id[i]),"code_lables":code_lables[i],"code_reply":code_reply}
                generation_json.append(item)
            
                        
    json.dump(
        generation_json,
        open(out_predict__path, 'w'),
        indent=4,
        ensure_ascii=False
    )

def get_instruction1(input, language, is_test = False):
        instruction =B_SYS + f"Give you a Programming problem,please Provide answers in {language}.  Wrap your code answer using ```."+ E_SYS + input
        if is_test:
            text = f"{B_INST} {(instruction).strip()} {E_INST}"
       
        return text 
 
def get_instruction2(input, language, is_test = False):
        instruction =B_SYS + f"Give you a piece of {language} code, please continue to write the unfinished function.  Wrap your code answer using ```."+ E_SYS + input
        if is_test:
            text = f"{B_INST} {(instruction).strip()} {E_INST}"
        return text 
def get_instruction3(input, language, is_test = False):
        instruction =B_SYS + f"Here is an incomplete code, you need to complete. Wrap your code answer using ```, Your code must include a complete implementation of the 'Solution' class with exactly one function in the class."+ E_SYS + input
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
    for item in tqdm(data_list):
        inputs = item['prompt']
        text = get_instruction2(inputs,language="Java",is_test=True)
        text = tokenizer(text,return_tensors='pt')
        input_ids = text['input_ids'].cuda()
        task_id = item['task_id']
        batch_output = evaluate(model, tokenizer,input_ids, pad_token_id = 0)
        
        for i in range(len(batch_output)):
            code_reply = filte_code_for_humaneval(batch_output[i])
            #print(code_reply)
        
            item = {"task_id":task_id,"generation":code_reply,"prompt":inputs}
            generation_json.append(item)
            with open("./out_predict/humaneval_java_base_model_test.jsonl", 'w') as file:
                for item in generation_json:
                    json.dump(item, file)
                    file.write('\n')
    with open("./out_predict/humaneval_basemodel_P1.jsonl", 'w') as file:
        for item in generation_json:
            json.dump(item, file)
            file.write('\n')
        
def add_prompt(data_path,result_data_path):
    data_list = load_json_data(data_path)
    result_data_path = load_json_data(result_data_path)
    generation_json = []
    indx = 0
    for item in tqdm(data_list):
        inputs = item['prompt']
        result_data_path[indx]['prompt'] = inputs
        #print(result_data_path[indx])
        #input()
        generation_json.append(result_data_path[indx])
        indx += 1
    with open("./out_predict/humeval_java_base_finally.jsonl", 'w') as file:
        for item in generation_json:
            json.dump(item, file)
            file.write('\n')

def re_generate_json(data_path):
    data_list = load_json_data(data_path)
    generation_json = []
    indx = 0
    for item in tqdm(data_list):
        item['generation'] = filte_code(item['generation'])
        generation_json.append(item)
        indx += 1
    with open("./out_predict/humeval_java_base_use_test.jsonl", 'w') as file:
        for item in generation_json:
            json.dump(item, file)
            file.write('\n')       
    
if __name__ == "__main__":
    #print("start")
    #predict_eval_test_data(out_predict__path = "./out_predict/base_Long50.json")
    generate_humeval_data("./humaneval/humaneval_java.jsonl")
    #add_prompt("./humaneval/humaneval_java.jsonl","./out_predict/humeval_java_base_fix.jsonl")
    #text = "[INST] <<SYS>>\nGive you a Programming problem,please Provide answers in Java\n<</SYS>>\n\nimport java.util.*;\nimport java.lang.*;\n\nclass Solution {\n    /**\n    Return length of given string\n    >>> strlen(\"\")\n    0\n    >>> strlen(\"abc\")\n    3\n     */\n    public int strlen(String string) { [/INST]  To get the length of a string in Java, you can use the `length()` method of the `String` class.\n\nHere's an example:\n```\nString str = \"Hello, World!\";\nint length = str.length();\nSystem.out.println(length); // Output: 12\n```\nAlternatively, you can also use the `length()` method of the `StringBuilder` class, which is more efficient for large strings:\n```\nStringBuilder sb = new StringBuilder(\"Hello, World!\");\nint length = sb.length();\nSystem.out.println(length); // Output: 12\n```\nNote that the `length()` method returns the number of characters in the string, not the number of bytes. If you need to get the number of bytes, you can use the `getBytes()` method of the `String` class, which returns a `byte[]` array:\n```\nString str = \"Hello, World!\";\nbyte[] bytes = str.getBytes();\nSystem.out.println(bytes.length); // Output: 13\n```\nIn this case, the `length()` method returns the number of bytes in the `byte[]` array, which is 13 in this case."
    #text1 = filte_code(text)
    
    
    #re_generate_json("./out_predict/humeval_java_base_2.jsonl")
    