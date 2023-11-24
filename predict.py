from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import os
from arguments import train_config
import torch
import transformers 
from transformers import LlamaTokenizer,Trainer
from load_data import processClass
from model import PreferCodeLlama
from model_aug import PreferAugCodeLlama
from model_aug_t import PreferAugTCodeLlama
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from train_utils import load_model_checkpoint 
from memory_utils import MemoryTrace
from tqdm import tqdm  
PATH_TO_CONVERTED_WEIGHTS = "../CodeLlama-7b-Instruct-hf"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def print_trainable_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            print(f"Parameter name: {name}, Shape: {param.shape}")
    print(f"Total Trainable Parameters: {total_params}")

    
    
def generate(model, tokenizer, args, batch):
    # Turn on evaluation mode which disables dropout.
    
    stop_token = tokenizer.eos_token_id
    max_total_seq_len = args.max_total_seq_len
    max_generate_length = args.max_generate_length
    with torch.no_grad():
        
        user_id = batch['user_id'].cuda()
        batch_size = user_id.size(0)
        input_ids = batch['input_ids'].cuda() #test应该只有问题而没有回答
        attention_mask = batch['attention_mask'].cuda()
        #labels = batch['labels'].cuda()
        lengths = attention_mask.sum(dim=1)
        max_prompt_len = torch.max(lengths)
        min_prompt_len = torch.min(lengths).item()
        pad_token_id = tokenizer.pad_token_id
        total_len = min(max_total_seq_len, max_generate_length + max_prompt_len)
        
        padding_size = total_len - input_ids.size(1)
        pad_tokens = torch.full((batch_size, padding_size), pad_token_id, dtype=torch.long, device="cuda") 
        tokens = torch.cat([input_ids, pad_tokens], dim=1)
        
        #pad_mask = torch.full((batch_size, padding_size), 0, dtype=torch.long, device="cuda")
        #input_text_mask = torch.cat([attention_mask, pad_mask], dim=1)
        
        input_text_mask = tokens != pad_token_id
        prev_pos = 0
        stop_reached = torch.tensor([False] * batch_size, device="cuda")
        #model forward 自回归生成tokens
        
        #print(total_len)
        #print(max_prompt_len)
        
        in_tokens = tokens[:, 0:min_prompt_len]

        kvcache = None
        for cur_pos in range(min_prompt_len, total_len):
            #print(cur_pos)
            if args.choose_model_name == "perfer_Base":
                modeling_outputs = model(user_id = user_id,input_ids = in_tokens,attention_mask = None,past_key_values=kvcache) #think
                logits = modeling_outputs.logits
                kvcache = modeling_outputs.past_key_values
                
                if args.temperature > 0:
                    probs = torch.softmax(logits[:, -1] / args.temperature, dim=-1)
                    next_token = sample_top_p(probs, args.top_p)
                else:
                    next_token = torch.argmax(logits[:, -1], dim=-1)
            elif args.choose_model_name =="perfer_Aug":
                modeling_outputs, P_final = model(user_id = user_id,input_ids = in_tokens,attention_mask = None,past_key_values=kvcache) #think
                #logits = modeling_outputs.logits
                kvcache = modeling_outputs.past_key_values
                next_token = torch.argmax(P_final[:, -1], dim=-1)
            elif args.choose_model_name =="perfer_AugT":
                modeling_outputs, P_final = model(user_id = user_id,input_ids = in_tokens,attention_mask = None,past_key_values=kvcache) #think
                #logits = modeling_outputs.logits
                kvcache = modeling_outputs.past_key_values
                next_token = torch.argmax(P_final[:, -1], dim=-1)  
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            ) #小于输入长度时，选择tokens[:, cur_pos]为next_token
            
            in_tokens = next_token.unsqueeze(1)
            
            tokens[:, cur_pos] = next_token
            stop_reached |= (~input_text_mask[:, cur_pos]) & (next_token == stop_token)
            
            #prev_pos = cur_pos
            if all(stop_reached):
                break
        #过滤生成token中的提问
        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = lengths[i].item()
            toks = toks[start : start + max_generate_length]
            
            # cut to stop token if present
            if stop_token in toks:
                stop_idx = toks.index(stop_token)
                toks = toks[:stop_idx]
                
            out_tokens.append(toks)

        return out_tokens
    

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


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
    #print(text)
    #input()
    #text = text.split(E_INST)[1]
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
    
    
def predict(model, train_config, test_dataloader, tokenizer):
    """
    Evaluates the model on the given dataloader
    
    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions
    
    Returns: eval_ppl, eval_epoch_loss
    """
     
    model.eval()
    
    
    generation_json = []
    for step, batch in enumerate(tqdm(test_dataloader,colour="green", desc="predict Epoch", dynamic_ncols=True)):
        with MemoryTrace() as memtrace:
            generation_tokens = generate(model, tokenizer, train_config, batch)
            user_id = batch['user_id']
            input_ids = batch['input_ids'] #test应该只有问题而没有回答
            code_lables = batch['code_labels']
            problem_id  = batch['problem_id'] 
            #for i in range(train_config.per_device_test_batch_size):
            #    generation_token = generation_tokens[i]
            code_generations = [tokenizer.decode(t) for t in generation_tokens]
            
            for i in range(len(code_generations)):
                code_generation = code_generations[i]
                code_generation = filte_code(code_generation)
                item = {"user_id":str(user_id[i].item()),"problem_id":problem_id[i],"code_lables": code_lables[i],"code_reply":str(code_generation)}
                generation_json.append(item)
                if train_config.choose_model_name != "perfer_Base" and len(generation_json) %10 == 0:
                    json.dump(
                        generation_json,
                        open(train_config.predict_dirs, 'w'),
                        indent=4,
                        ensure_ascii=False
                    )
         
    json.dump(
        generation_json,
        open(train_config.predict_dirs, 'w'),
        indent=4,
        ensure_ascii=False
    )
    
def load_json_data(data_path):
    #print("loading text-score dataset from: \n   {}".format(data_path))
    data_list = []
    with open(data_path, 'r') as file:
        for line in file:
            # 逐行解析JSON对象
            data = json.loads(line)
            data_list.append(data)
    return data_list

def get_instruction(input, answer,language, is_test = False):
        instruction =B_SYS + f"Give you a Programming problem,please Provide answers in {language}"+ E_SYS + input
        if is_test:
            text = f"{B_INST} {(instruction).strip()} {E_INST}"
        else:
            text = f"{B_INST} {(instruction).strip()} {E_INST} ```\n{(answer).strip()}```\n </s>"
        return text 
          
def get_instruction2(input, answer,language, is_test = False):
        instruction =B_SYS + f"Give you a piece of {language} code, please continue to write the unfinished function "+ E_SYS + input
        if is_test:
            text = f"{B_INST} {(instruction).strip()} {E_INST}"
        else:
            text = f"{B_INST} {(instruction).strip()} {E_INST} ```\n{(answer).strip()}```\n </s>"
        return text 
def get_instruction3(input, answer,language, is_test = False):
        instruction =B_SYS + f"Here is an incomplete code, you need to complete. Wrap your code answer using ```, Your code must include a complete implementation of the 'Solution' class with exactly one function in the class."+ E_SYS + input
        if is_test:
            text = f"{B_INST} {(instruction).strip()} {E_INST}"
        return text   
    
def human_eval():
    
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0] 
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name_or_path)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )        
    args.idnum = 50
    print(f"user number = {args.idnum}")
    # inint model
    #---------------------------------------------------------------------------------
    if args.choose_model_name == "perfer_Base":
        model = PreferCodeLlama(args)
    elif args.choose_model_name =="perfer_Aug":
        model = PreferAugCodeLlama(args)
    elif args.choose_model_name =="perfer_AugT":
        model = PreferAugTCodeLlama(args)
    model = load_model_checkpoint(model, 0, args)
    model = torch.nn.DataParallel(model)
    on_gpu = all(param.is_cuda for param in model.parameters())
    if on_gpu:
        print("模型在 GPU 上运行。")
    else:
        print("模型在 CPU 上运行。")
    print_trainable_parameters(model)  
    model.eval()
    
    test_list = load_json_data("./data/humaneval_java.jsonl")
    print(args.human_eval_out_path)
    generation_json = []
    user_id = -1
    for step, item in enumerate(tqdm(test_list,colour="green", desc="predict Epoch", dynamic_ncols=True)):
        with MemoryTrace() as memtrace:
            #print(item)
            user_id = (user_id + 1)% args.idnum
            batch = {}
            batch['user_id'] = [user_id]
            encoded_inputs = tokenizer(get_instruction3(item['prompt'],"","Java", True),padding=True,return_tensors='pt')
            #print(get_instruction(item['prompt'],"",True))
            #input()
            batch['input_ids'] = encoded_inputs["input_ids"]
            batch['attention_mask'] = encoded_inputs["attention_mask"]
            batch_item = {
                "user_id": torch.tensor(batch['user_id']).long(),
                "input_ids": batch['input_ids'],
                "attention_mask": batch['attention_mask'],
            }
            #print(batch)
            #input()
            generation_tokens = generate(model, tokenizer, args, batch_item)
               
            code_generations = [tokenizer.decode(t) for t in generation_tokens]
            
            for i in range(len(code_generations)):
                code_generation = code_generations[i]
                code_generation = filte_code(code_generation)
                item_code = {"user_id":str(user_id),"task_id":item['task_id'],"generation": code_generation,"prompt":item['prompt']}
                generation_json.append(item_code)
            with open(args.human_eval_out_path, 'w') as file:
                for item in generation_json:
                    json.dump(item, file)
                    file.write('\n')    
    with open(args.human_eval_out_path, 'w') as file:
        for item in generation_json:
            json.dump(item, file)
            file.write('\n')     
   
   
    
        
    

def main():

    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.cuda.empty_cache()
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0]
    
    #print(args)
    
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name_or_path)
    print(tokenizer.truncation_side)
    #tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name_or_path, model_max_length=args.max_length)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )    
    
    # load data
    #---------------------------------------------------------------------------------  
    proc = processClass()
    train_data_set = proc.get_train_dataset(args,tokenizer, is_test = False)
    eval_dataset_set = proc.get_eval_datasets(args,tokenizer,  is_test = False)
    test_data_set = proc.get_test_datasets(args,tokenizer,is_test = True)    
    args.idnum = proc.idnum
    print(f"user number = {args.idnum}")
    test_dataloader = DataLoader(test_data_set , batch_size=args.per_device_test_batch_size, collate_fn=test_data_set.collate_batch, num_workers=4)
    #print(tokenizer.pad_token_id)

    # inint model
    #---------------------------------------------------------------------------------
    if args.choose_model_name == "perfer_Base":
        model = PreferCodeLlama(args)
    elif args.choose_model_name =="perfer_Aug":
        model = PreferAugCodeLlama(args)
    elif args.choose_model_name =="perfer_AugT":
        model = PreferAugTCodeLlama(args)
    model = load_model_checkpoint(model, 0, args)
    model = model.cuda()
   
    #model = torch.nn.DataParallel(model)
    
    on_gpu = all(param.is_cuda for param in model.parameters())
    if on_gpu:
        print("模型在 GPU 上运行。")
    else:
        print("模型在 CPU 上运行。")
    print_trainable_parameters(model)  
    predict(model, args, test_dataloader,tokenizer)
    # build trainer
    #---------------------------------------------------------------------------------

    
if __name__ == "__main__":
    main()
    #human_eval()