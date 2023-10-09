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
from train_utils import train
def print_trainable_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            if "user_embeddings_list" not in name:
                print(f"Parameter name: {name}, Shape: {param.shape}")
    print(f"Total Trainable Parameters: {total_params}")
    
def main():
    #os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #os.environ['TORCH_USE_CUDA_DSA'] = "1"

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
    idnum = proc.idnum
    print(idnum)
    print("--------")
    eval_dataset_set = proc.get_eval_datasets(args,tokenizer)
    idnum = proc.idnum
    print(idnum)
    print("--------")
    input()
    idnum = proc.idnum
    train_dataloader = DataLoader(train_data_set , batch_size=args.per_device_train_batch_size, collate_fn=train_data_set.collate_batch, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset_set , batch_size=args.per_device_eval_batch_size, collate_fn=train_data_set.collate_batch,  shuffle = False, num_workers=4)
    
    
    args.idnum = idnum
    # inint model
    #---------------------------------------------------------------------------------
    model = PreferCodeLlama(args)
    #model = load_model_checkpoint(model,args)
    model = model.cuda()
    print_trainable_parameters(model)  
    # build trainer
    #---------------------------------------------------------------------------------

    if args.do_train:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
        results = train(
            model,
            train_dataloader,
            eval_dataloader,
            tokenizer,
            optimizer,
            scheduler,
            args.gradient_accumulation_steps,
            args,
            fsdp_config =  None,
            local_rank = None,
            rank = None,
        )  
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
if __name__ == "__main__":
    main()