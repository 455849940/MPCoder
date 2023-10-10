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
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from train_utils import load_model_checkpoint 
from llama_recipes.utils.memory_utils import MemoryTrace
from tqdm import tqdm 
def print_trainable_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            if "user_embeddings_list" not in name:
                print(f"Parameter name: {name}, Shape: {param.shape}")
    print(f"Total Trainable Parameters: {total_params}")

    
    
def generate(model, tokenizer, args, batch):
    # Turn on evaluation mode which disables dropout.
    
    stop_token = tokenizer.eos_token_id
    max_total_seq_len = args.max_total_seq_len
    max_generate_length = args.max_generate_length
    with torch.no_grad():
        print("start generate")
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
        print("yes")
        print(total_len)
        print(max_prompt_len)
        print("-------")
        in_tokens = tokens[:, 0:min_prompt_len]

        kvcache = None
        for cur_pos in range(min_prompt_len, total_len):
            #print(cur_pos)
            modeling_outputs = model.forward(user_id = user_id,input_ids = in_tokens,attention_mask = None,past_key_values=kvcache) #think
            logits = modeling_outputs.logits
            kvcache = modeling_outputs.past_key_values
            
            if args.temperature > 0:
                probs = torch.softmax(logits[:, -1] / args.temperature, dim=-1)
                next_token = sample_top_p(probs, args.top_p)
            else:
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
            #for i in range(train_config.per_device_test_batch_size):
            #    generation_token = generation_tokens[i]
            code_generations = [tokenizer.decode(t) for t in generation_tokens]
            for i in range(len(code_generations)):
                code_generation = code_generations[i]
                item = {"user_id":str(user_id[i]),"problem":str(tokenizer.decode(input_ids[i])),"code_reply":str(code_generation)}
                generation_json.append(item)
    json.dump(
        generation_json,
        open("./out_predict/result.json", 'w'),
        indent=4,
        ensure_ascii=False
    )
        
                
   
    
        
    

def main():
    
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0]
    
    print(args)
    
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
    train_data_set = proc.get_train_dataset(args,tokenizer)
    test_data_set = proc.get_test_datasets(args,tokenizer,is_test = True)    
    args.idnum = proc.idnum
    test_dataloader = DataLoader(test_data_set , batch_size=args.per_device_test_batch_size, collate_fn=test_data_set.collate_batch, num_workers=4)
    #print(tokenizer.pad_token_id)

    # inint model
    #---------------------------------------------------------------------------------
    model = PreferCodeLlama(args)
    model = load_model_checkpoint(model,args)
    model = model.cuda()
    print_trainable_parameters(model)  
    predict(model, args, test_dataloader,tokenizer)
    # build trainer
    #---------------------------------------------------------------------------------

    
if __name__ == "__main__":
    main()