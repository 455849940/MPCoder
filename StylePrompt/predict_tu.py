from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import os
from feature_arguments import train_config
from load_feature_data import FeatureProcessClass
import torch
import transformers 
from transformers import LlamaTokenizer,Trainer
from load_user_data import processClass
from FeatureModel_M import PreferFeatureCodeLlama_M
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
        forwardChoose = model.forwardChoose
        if forwardChoose == 3:
            hiddens_states = model(user_id = user_id,input_ids = in_tokens,attention_mask = None,select_mask = in_select_mask,past_key_values=kvcache)


        return hiddens_states.cpu().numpy().tolist()
    
    
    
def predict(model, train_config, test_dataloader, tokenizer):
    
    
    model.eval()
    
    generation_json = []
    for step, batch in enumerate(tqdm(test_dataloader,colour="green", desc="predict Epoch", dynamic_ncols=True)):
        with MemoryTrace() as memtrace:
            generation_hidden_states = generate(model, tokenizer, train_config, batch)
            user_id = batch['user_id']
            #input_ids = batch['input_ids'] #test只有问题而没有回答
            code_lables = batch['code_labels']
            problem_id  = batch['problem_id'] 
            
            for i in range(len(generation_hidden_states)):
                code_generation = generation_hidden_states[i]
                item = {"user_id":str(user_id[i].item()),"problem_id":problem_id[i],"code_reply":str(code_generation)}
                generation_json.append(item)
                
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


def main():

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    torch.cuda.empty_cache()
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0]
    
    #print(args)
    if args.is_predict == False : return 
    
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name_or_path)
    
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )    
    #add Tokens
    print(len(tokenizer))
    Feature_proc = FeatureProcessClass(tokenizer)
    print(len(tokenizer))
    # load data
    #---------------------------------------------------------------------------------  
    proc = processClass(args.user_style_data_path)
    train_data_set = proc.get_train_dataset(args,tokenizer, is_test = False)
    eval_dataset_set = proc.get_eval_datasets(args,tokenizer,  is_test = False)
    test_data_set = proc.get_test_datasets(args,tokenizer,is_test = True)    
    args.idnum = proc.idnum
    #args.idnum = 1121
    print(f"user number = {args.idnum}")
    test_dataloader = DataLoader(test_data_set , batch_size=args.per_device_test_batch_size, collate_fn=test_data_set.collate_batch, num_workers=4)
    

    # inint model
    #---------------------------------------------------------------------------------
    
    model = PreferFeatureCodeLlama_M(args)
    model = load_model_checkpoint(model, 0, args.output_dir2)
    model.set_forwardChoose(args.forwardChoose2)
    print(">>>model.forwardChoose")
    print(model.forwardChoose)
    model = model.cuda()
    #model = torch.nn.DataParallel(model)
    
    on_gpu = all(param.is_cuda for param in model.parameters())
    if on_gpu:
        print("模型在 GPU 上运行。")
    else:
        print("模型在 CPU 上运行。")
    #print_trainable_parameters(model)  
    predict(model, args, test_dataloader,tokenizer)
    # build trainer
    #---------------------------------------------------------------------------------

    
if __name__ == "__main__":
    main()
    #human_eval()