import json
from tqdm import tqdm

from copy import deepcopy

import numpy as np
from arguments import train_config
import torch
from torch.utils.data import Dataset
from transformers import LlamaTokenizer
import transformers
from itertools import chain
from transformers import DataCollator
from tqdm import tqdm
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
class TextRewardDataset(Dataset):
        def __init__(self, data,tokenizer):
            self.data = data 
            self.tokenizer = tokenizer
        def __getitem__(self, index):
            return self.data[index]

        def __len__(self,):
            return len(self.data)
        
        def collate_batch(self, batch):
            id_batch = []
            input_ids_batch = []
            attention_mask_batch = []
            labels_batch = []
            instruction_list = []
            problem_id_batch = []
            for item in batch:
                id_batch.append(item['user_id'])
                instruction_list.append(item['input'])
                labels_batch.append(item['code']) 
                problem_id_batch.append(item['problem_id'])
            encoded_inputs = self.tokenizer(instruction_list,padding=True,return_tensors='pt')
            input_ids_batch = encoded_inputs["input_ids"]
            attention_mask_batch = encoded_inputs["attention_mask"]
    
            
            return {
                "user_id": torch.tensor(id_batch).long(),
                "input_ids": input_ids_batch,
                "attention_mask": attention_mask_batch,
                "code_labels": labels_batch,
                "problem_id": problem_id_batch
            }
            
    
class processClass:
    def __init__(self,idnum = 0):
            self.idnum = idnum
            self.id2vidmap = dict()
            self.vid2idmap = dict()  
                
    def get_instruction(self, input, answer,language, is_test = False):
        instruction =B_SYS + f"Give you a Programming problem,please Provide answers in {language}"+ E_SYS + input
        if is_test:
            text = f"{B_INST} {(instruction).strip()} {E_INST}"
        else:
            text = f"{B_INST} {(instruction).strip()} {E_INST} ```\n{(answer).strip()}```\n </s>"
        return text 
    
    
    def prepare_data_item(self, language, items, problem_content,tokenizer=None, padding=False, is_test = False):
        data_list = []
    
        for i in range(0,len(items)):
            new_items = {}
            user_id = items[i]['user_id']
            if user_id not in self.id2vidmap:
                self.id2vidmap[user_id] = self.idnum
                self.vid2idmap[self.idnum] = user_id
                self.idnum += 1
            
            
            new_items['code'] = items[i]['code'].strip('\n') #去掉前后多余的空行
            user_vir_id = self.id2vidmap[user_id]
            new_items['user_id'] = user_vir_id
            
            new_items['problem_id'] = items[i]['problem_id']
            
            new_items['input'] = self.get_instruction(problem_content[i], new_items['code'], language,is_test) 
            text = tokenizer(new_items['input'],return_tensors='pt')
            if is_test is False and text['input_ids'].shape[1] > 2048:
                continue
            data_list.append(new_items)
            
            
        return  data_list   

          
    def load_json_data(self,data_path):
        #print("loading text-score dataset from: \n   {}".format(data_path))
        with open(data_path, 'r') as f:
            data_list = json.load(f)
            #data_list = sorted(data_list, key=lambda x: x["user_id"])
        return data_list

    #获取部分还是全部数据
    def get_data_iter(self,data_list, debug=False, is_test=False):
        if debug:
            data_size = len(data_list)
            if is_test:
                up_data_size = data_size
            else :
                up_data_size = 100
            data_list = [data_list[i] for i in range(min(int(0.1*data_size), up_data_size))]

        #if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
        #    return tqdm(data_list)
        #else:
        return data_list
        
    #可能根据test和train的不同修改成对应的数据内容 字段名可以一致但内容不同
    def load_text_score_dataset(self,language, problem_path, data_path, tokenizer=None, debug=False, padding=False, batch_size = 1,is_test=False,rank = 0):
        if rank == 0:
            print("loading text-score dataset from: \n   {}".format(data_path))

        if data_path[-4:] == 'json':
            data_list = self.load_json_data(data_path)
        problem_list = self.load_json_data(problem_path)
        
        if is_test == False:
            data_list = sorted(data_list, key=lambda x: x["user_id"]) #按id排序训练
            
        problem_map = dict()
        for item in problem_list:
            problem_map[item['id']] = item['english_content']
        
        outputs = []
        data_list = self.get_data_iter(data_list, debug=debug, is_test=is_test)
        data_list_len = len(data_list)
        for i in range(0, len(data_list), batch_size):
            items = data_list[i:min(i+batch_size, data_list_len)]
            problem_list =  [problem_map[item['problem_id']] for item in items]
            new_item = self.prepare_data_item(language, items, problem_content = problem_list,tokenizer=tokenizer, padding=padding, is_test = is_test)
            
            if new_item is not None:
                outputs.append(new_item)
            
            
        outputs = list(chain.from_iterable(outputs))
        if rank == 0:
            print("finished processing {}  data. in {}".format(len(outputs), data_path))
        return outputs

    def get_train_dataset(self,args, tokenizer, is_test = False, rank = 0):    
        all_train_data = []
        for train_data_path in args.train_data_path:
            train_data = self.load_text_score_dataset(
                language = args.language,
                problem_path=args.problem_path,
                data_path=train_data_path,
                tokenizer=tokenizer, 
                debug=args.debug_mode,
                padding=not args.per_device_train_batch_size == 1,
                batch_size = args.per_device_train_batch_size,
                is_test = is_test,
                rank = rank
            )
            all_train_data.extend(train_data)
        if args.debug_mode:
            print(f">>> check tokenized data:")        
            print(f">>> {all_train_data[0]}")
        train_set = TextRewardDataset(all_train_data, tokenizer)
        return train_set

    def get_eval_datasets(self,args, tokenizer, is_test = False, rank = 0):
        all_eval_data = []
        for data_path in args.eval_data_path:
            eval_data_list = self.load_text_score_dataset(
                language = args.language,
                problem_path=args.problem_path,
                data_path=data_path,
                tokenizer=tokenizer,
                debug=args.debug_mode,
                padding=not args.per_device_eval_batch_size == 1,
                batch_size = args.per_device_eval_batch_size,
                is_test = is_test,
                rank = rank
            )
            all_eval_data.extend(eval_data_list)
        eval_dataset = TextRewardDataset(eval_data_list, tokenizer)    
        return eval_dataset
    
    def get_test_datasets(self,args, tokenizer, is_test = True):
        all_test_data = []
        for data_path in args.test_data_path:
            test_data_list = self.load_text_score_dataset(
                language = args.language,
                problem_path=args.problem_path,
                data_path=data_path,
                tokenizer=tokenizer,
                debug=args.debug_mode,
                padding=not args.per_device_test_batch_size == 1,
                batch_size = args.per_device_test_batch_size,
                is_test = is_test
            )
            all_test_data.extend(test_data_list)
        test_dataset = TextRewardDataset(test_data_list, tokenizer)    
        return test_dataset
      
if __name__ == "__main__":
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0]
    args.debug_mode = True
    #print(args)
    
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name_or_path)
    print(tokenizer.padding_side)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    tokenizer.pad_token_id = 0
    prco = processClass()
    train_data_set = prco.get_train_dataset(args,tokenizer)
    print(train_data_set[3])
    print(train_data_set[4])