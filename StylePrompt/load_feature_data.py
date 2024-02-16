import json
from tqdm import tqdm

from copy import deepcopy

import numpy as np
from feature_arguments import train_config
import torch
from torch.utils.data import Dataset
from transformers import LlamaTokenizer
import transformers
from itertools import chain
from transformers import DataCollator
from tqdm import tqdm
import os
import sys
# 获取当前文件所在的文件夹路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级文件夹路径
parent_dir = os.path.dirname(current_dir)
# 将上一级文件夹路径加入到sys.path中
sys.path.append(parent_dir)
from configs.fsdp import fsdp_config

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
class TextRewardDataset(Dataset):
        def __init__(self, data,tokenizer):
            self.data = data 
            self.tokenizer = tokenizer
            self.up_num = 32016
        def __getitem__(self, index):
            return self.data[index]

        def __len__(self,):
            return len(self.data)
        
        def collate_batch(self, batch):
   
            input_ids_batch = []
            attention_mask_batch = []
            instruction_list = []
            for item in batch:
                instruction_list.append(item['input'])

            encoded_inputs = self.tokenizer(instruction_list,padding=True,return_tensors='pt')
            input_ids_batch = encoded_inputs["input_ids"]
            attention_mask_batch = encoded_inputs["attention_mask"]
            select_mask_batch = torch.where(input_ids_batch > self.up_num, input_ids_batch-self.up_num, torch.tensor(0, dtype=input_ids_batch.dtype))
            
            
            return { 
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "select_mask": select_mask_batch
            }
            
    
class FeatureProcessClass:
    def __init__(self, tokenizer):

        self.style_list = ['RightCurly','SeparatorWrap','NoLineWrapCheck', 'AvoidStarImportCheck', 'OneTopLevelClassCheck',
        'EmptyLineSeparatorCheck', 'WhitespaceAroundCheck', 'GenericWhitespaceCheck',
        'OperatorWrapCheck','LineLengthCheck','LeftCurlyCheck', 'EmptyBlockCheck',
        'NeedBracesCheck', 'IndentationCheck', 'MultipleVariableDeclarationsCheck',
        'OneStatementPerLineCheck','UpperEllCheck', 'ModifierOrderCheck', 
        'FallThroughCheck','MissingSwitchDefaultCheck', 
        'TypeNameCheck', 'MethodNameCheck','MemberNameCheck', 'ParameterNameCheck', 'LocalVariableNameCheck']  
        self.style_meaning_map = {
            'RightCurly':"Do not put ‘}’ on its own line.",
            'SeparatorWrap':"Do not break after ‘,’ but before ‘.’.",
            'NoLineWrapCheck':"break import and package lines.",
            'AvoidStarImportCheck':"import statements that use the * notation.",
            'OneTopLevelClassCheck':"Do not put a top class in its own file.",
            'EmptyLineSeparatorCheck':"Do not use a blank line after header, package, import lines, class, methods,fields, static, and instance initializers.", 
            'WhitespaceAroundCheck':"Do not use a space between a reserved word and its follow-up bracket,e.g., if(.", 
            'GenericWhitespaceCheck':"Use a space before the definition of generic type, e.g., List <.",
            'OperatorWrapCheck':"Break after ‘=’ but after other binary operators.",
            'LineLengthCheck':"The line length exceeds 100 characters",
            'LeftCurlyCheck':"Do not put ‘{’ on the same line of code.", 
            'EmptyBlockCheck':"Have empty block for control statements.",
            'NeedBracesCheck':"Do not use braces for single control statements.",
            'IndentationCheck':"Controls the indentation between comments and surrounding code.", 
            'MultipleVariableDeclarationsCheck':"Not every  variable declaration is in its own statement and on its own line.",
            'OneStatementPerLineCheck':"there is not only one statement per line.",
            'UpperEllCheck':"long constants are defined with an upper ell. That is 'l' and not 'L'.", 
            'ModifierOrderCheck':"Do not follow the order: public, protected, private, abstract, default,static,final, transient, volatile, synchronized, native, strictfp.", 
            'FallThroughCheck':"Do not put a fall-through comment in a switch If a ‘case’ has no break,return, throw, or continue.",
            'MissingSwitchDefaultCheck':"switch statement does not has a default clause.", 
            'TypeNameCheck':"Type name is not in UpperCamelCase.", 
            'MethodNameCheck':"Method name is not in lowerCamelCase.",
            'MemberNameCheck':"Member name is not in lowerCamelCase.",
            'ParameterNameCheck':"Parameter name is not in lowerCamelCase.", 
            'LocalVariableNameCheck':" Local variable name is not in lowerCamelCase."
        }
        tokenizer.add_tokens(self.style_list)
        
    def expand_tokenizer(self, tokenizer):
        tokenizer.add_tokens(self.style_list)
    
     
    def get_feature_instruction(self, code1, code2,Style_list1, Style_list2, addtional_feature, is_test = False):
        input = f"code1:```\n{code1}```\nlist1:{Style_list1}\ncode2:```\n{code2}```\nlist2:{Style_list2}\n"
        instruction =B_SYS + f"You are given two pieces of code, code1 and code2, along with their corresponding lists of style conventions, list1 and list2. Please identify and explain the style conventions in list2 that are not present in list1."+ E_SYS + input
        
        explan = self.style_meaning_map[addtional_feature]
        answer = f"{addtional_feature} is present in list2 but not in list1; the style convention of {addtional_feature} indicates '{explan}'"
        if is_test:
            text = f"{B_INST} {(instruction).strip()} {E_INST}"
        else:
            text = f"{B_INST} {(instruction).strip()} {E_INST} {(answer).strip()} </s>"
        return text 
    
    def get_feature_instruction_NResidual(self, code1, code2,Style_list1, Style_list2, addtional_feature, is_test = False):
        input = f"code:```\n{code1}```\n"
        instruction =B_SYS + f"You are given one piece of code along with their corresponding style convention. Please identify and explain the style convention."+ E_SYS + input
        
        explan = self.style_meaning_map[addtional_feature]
        answer = f"{addtional_feature} is present in code; the style convention of {addtional_feature} indicates '{explan}'"
        if is_test:
            text = f"{B_INST} {(instruction).strip()} {E_INST}"
        else:
            text = f"{B_INST} {(instruction).strip()} {E_INST} {(answer).strip()} </s>"
        return text 
    
    def prepare_feature_data_item(self, items, tokenizer=None,  is_test = False):
        data_list = []
      
        for i in range(0,len(items)):
            new_items = {}

            code1 = items[i]['code1'].strip('\n') #去掉前后多余的空行
            code2 = items[i]['code2'].strip('\n') #去掉前后多余的空行  
            list1 = items[i]['StyleFeature_list']   
            Style_list1 = []
            for idx,value in enumerate(list1):
                if value > 0:
                    Style_list1.append(self.style_list[idx])
            Style_list2 = deepcopy(Style_list1)
    
            addtional_feature_idx = int(items[i]['addtional_feature'])
            addtional_feature = self.style_list[addtional_feature_idx]
            
            Style_list2.append(addtional_feature)
            
            new_items['input'] = self.get_feature_instruction(code1, code2,Style_list1, Style_list2, addtional_feature,is_test)
             
            text = tokenizer(new_items['input'],return_tensors='pt')
            if is_test is False and text['input_ids'].shape[1] > 2048: #
                continue
            data_list.append(new_items)
            
            
        return  data_list   

    
                 
    def load_json_data(self,data_path):
        with open(data_path, 'r') as f:
            data_list = json.load(f)
        return data_list

    #获取部分还是全部数据
    def get_data_iter(self,data_list, debug=False, is_test=False):
        if debug:
            data_size = len(data_list)
            if is_test:
                up_data_size = data_size
            else :
                up_data_size = 100
            data_list = [data_list[i] for i in range(max(1,min(int(0.1*data_size), up_data_size)))]

        return data_list
        
    
      

    def load_feature_dataset(self, data_path , tokenizer=None,padding=False,batch_size = 1, is_test=False, debug=False, rank = 0):
        if rank == 0:
            print("loading Feature dataset from: \n   {}".format(data_path))
                
        data_list = self.load_json_data(data_path)
        
       
        
        outputs = []
        
        data_list = self.get_data_iter(data_list, debug=debug, is_test=is_test)
        
        data_list_len = len(data_list)
        
        for i in range(0, len(data_list), batch_size):
            items = data_list[i:min(i+batch_size, data_list_len)]
           
            new_item = self.prepare_feature_data_item(items,tokenizer=tokenizer, is_test = is_test)
         
            if new_item is not None:
                outputs.append(new_item)
            
            
        outputs = list(chain.from_iterable(outputs))
        if rank == 0:
            print("finished processing {}  data. in {}".format(len(outputs), data_path))
        return outputs
        
    def get_dataset(self,args, tokenizer, choose,is_test = False, rank = 0):    
        all_train_data = []
        if choose == "train":
            path =  args.feature_train_data_path
        elif choose == "dev":
            path =  args.feature_dev_data_path
        for train_data_path in path:
            train_data = self.load_feature_dataset(
                data_path=train_data_path,
                tokenizer=tokenizer, 
                batch_size = args.per_device_train_batch_size,
                padding=not args.per_device_train_batch_size == 1,
                is_test = is_test,
                debug=args.debug_mode,
                rank = rank
            )
            all_train_data.extend(train_data)
        
     
        if args.debug_mode:
            print(f">>> check tokenized data:")        
            print(f">>> {all_train_data[0]}")
        train_set = TextRewardDataset(all_train_data, tokenizer)
        return train_set

    
    
   
      
if __name__ == "__main__":
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0]
    args.debug_mode = True
    print(args)
    
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name_or_path)
    print(tokenizer.padding_side)
    print(len(tokenizer))
    
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    print(len(tokenizer))
    proc = FeatureProcessClass(tokenizer)
    #proc.expand_tokenizer(tokenizer)
    words = tokenizer.convert_ids_to_tokens(32016)
    print(words)
    for i in range(0,25):
        words = tokenizer.convert_ids_to_tokens(32017+i)
        print(words)
    #print(len(tokenizer))
    train_data_set = proc.get_dataset(args,tokenizer,  is_test = False,rank = 0)
    #print(train_data_set[0])
    print(">>>>>>>>>>>>>>>>>>")
    
    encoded_inputs = tokenizer([train_data_set[0]['input'],train_data_set[1]['input']],padding=True,return_tensors='pt')
    input_ids_batch = encoded_inputs["input_ids"]
    attention_mask_batch = encoded_inputs["attention_mask"]
    x = 32016
    select_mask_batch = torch.where(input_ids_batch > x, input_ids_batch, torch.tensor(x, dtype=input_ids_batch.dtype))
    #select_mask_batch = torch.where(input_ids_batch > x, input_ids_batch - x, torch.tensor(x, dtype=input_ids_batch.dtype))
    print(select_mask_batch)
    words = tokenizer.decode(select_mask_batch[1], skip_special_tokens=True)
    print(words)
    # example_token = tokenizer.encode(train_data_set[0]['input'],add_special_tokens=True)
    # words = tokenizer.convert_ids_to_tokens(example_token)
    # print(words)
    # print(example_token)
    # words = tokenizer.decode(example_token, skip_special_tokens=True)
    # print(words)
    #words = tokenizer.convert_ids_to_tokens(example_token)
    #print(words)
    
    