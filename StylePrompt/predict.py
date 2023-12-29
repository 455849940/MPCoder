from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import os
from feature_arguments import train_config
import torch
import transformers 
from transformers import LlamaTokenizer,Trainer
from load_user_data import processClass
from FeatureModel import PreferFeatureCodeLlama
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from feature_train_utils import load_model_checkpoint 
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
        select_mask = batch['select_mask'].cuda()
        #labels = batch['labels'].cuda()
        lengths = attention_mask.sum(dim=1)
        max_prompt_len = torch.max(lengths)
        min_prompt_len = torch.min(lengths).item()
        pad_token_id = tokenizer.pad_token_id
        total_len = min(max_total_seq_len, max_generate_length + max_prompt_len)
        
        padding_size = total_len - input_ids.size(1)
        pad_tokens = torch.full((batch_size, padding_size), pad_token_id, dtype=torch.long, device="cuda") 
        tokens = torch.cat([input_ids, pad_tokens], dim=1)
    
        input_text_mask = tokens != pad_token_id
        
        stop_reached = torch.tensor([False] * batch_size, device="cuda")
        #model forward 自回归生成tokens
        
        in_tokens = tokens[:, 0:min_prompt_len]
        
        #pad_tokens = torch.full((batch_size, padding_size), pad_token_id, dtype=torch.long, device="cuda") 
        #select_tokens = torch.cat([select_mask, pad_tokens], dim=1)
        in_select_mask = select_mask[:, 0:min_prompt_len]
        
        
        kvcache = None
        args.forwardChoose = model.forwardChoose
        for cur_pos in range(min_prompt_len, total_len):
            #print(cur_pos)
            # print(select_mask.shape)
            # print(in_tokens.shape)
            # print(in_select_mask.shape)
            # input()
            
            if args.forwardChoose == 1:
                modeling_outputs, P_final = model(user_id = user_id,input_ids = in_tokens,attention_mask = None,select_mask = in_select_mask,past_key_values=kvcache) #think
                #logits = modeling_outputs.logits
                kvcache = modeling_outputs.past_key_values
                next_token = torch.argmax(P_final[:, -1], dim=-1)  
            elif args.forwardChoose == 2:
                modeling_outputs = model(user_id = user_id,input_ids = in_tokens,attention_mask = None,select_mask = in_select_mask,past_key_values=kvcache) #think
                #logits = modeling_outputs.logits
                kvcache = modeling_outputs.past_key_values
                logits = modeling_outputs.logits
                next_token = torch.argmax(logits[:, -1], dim=-1)
                
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
    
    model.forwardChoose = 1
    model.eval()
    
    generation_json = []
    for step, batch in enumerate(tqdm(test_dataloader,colour="green", desc="predict Epoch", dynamic_ncols=True)):
        with MemoryTrace() as memtrace:
            generation_tokens = generate(model, tokenizer, train_config, batch)
            user_id = batch['user_id']
            input_ids = batch['input_ids'] #test只有问题而没有回答
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
                if len(generation_json) %10 == 0:
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
    



def main():

    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.cuda.empty_cache()
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0]
    
    #print(args)
    
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name_or_path)
    
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )    
    
    # load data
    #---------------------------------------------------------------------------------  
    proc = processClass(args.user_style_data_path)
    train_data_set = proc.get_train_dataset(args,tokenizer, is_test = False)
    eval_dataset_set = proc.get_eval_datasets(args,tokenizer,  is_test = False)
    test_data_set = proc.get_test_datasets(args,tokenizer,is_test = True)    
    #args.idnum = proc.idnum
    args.idnum = 50
    print(f"user number = {args.idnum}")
    test_dataloader = DataLoader(test_data_set , batch_size=args.per_device_test_batch_size, collate_fn=test_data_set.collate_batch, num_workers=4)
    

    # inint model
    #---------------------------------------------------------------------------------
    
    model = PreferFeatureCodeLlama(args)
    model = load_model_checkpoint(model, 0, args.output_dir2)
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