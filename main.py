from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import os
from arguments import train_config
import torch
import transformers 
from transformers import LlamaTokenizer,Trainer
from torch.utils.data import DistributedSampler
from load_data import processClass
from model import PreferCodeLlama
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from train_utils import train, clear_gpu_cache, setup_environ_flags, setup, get_policies
import torch.distributed as dist
from configs.fsdp import fsdp_config
from configs.wrapping import get_llama_wrapper
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
)
from torch.nn.parallel import DistributedDataParallel as DDP
def print_trainable_parameters(model):
    total_params = 0
    for name, param in model.module.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            print(f"Parameter name: {name}, Shape: {param.shape}")
    print(f"Total Trainable Parameters: {total_params}")
    
def main():
    dist.init_process_group("nccl")
    parser = transformers.HfArgumentParser(train_config)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.enable_fsdp:
        #setup()
        
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    if torch.distributed.is_initialized():
        print("yes")
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
    #print(args)
    
    tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name_or_path)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )    
    example_token = tokenizer.encode(" [INST] write a book")
    # print(example_token)
    # words = tokenizer.convert_ids_to_tokens(example_token)
    # print(words)
    # input()
    #bos_token = tokenizer.eos_token_id
    #print(bos_token)
    #words = tokenizer.convert_ids_to_tokens(bos_token)
    #print(words)
    #input()
    
    # load data
    #---------------------------------------------------------------------------------  
    proc = processClass()
    train_data_set = proc.get_train_dataset(args,tokenizer,  is_test = False,rank = rank)
    idnum = proc.idnum
    if not train_config.enable_fsdp or rank == 0:
        print(idnum)
        print("--------")
    eval_dataset_set = proc.get_eval_datasets(args,tokenizer,  is_test = False,rank = rank)
    idnum = proc.idnum
    if not train_config.enable_fsdp or rank == 0:
        print(idnum)
        print("--------")
    
    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            train_data_set,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            #shuffle=True,
        )
        if train_config.do_eval:
            val_sampler = DistributedSampler(
                eval_dataset_set,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )
    
    
    train_dataloader = DataLoader(train_data_set , batch_size=args.per_device_train_batch_size, collate_fn=train_data_set.collate_batch, num_workers=4,sampler=train_sampler if train_sampler else None)
    eval_dataloader = DataLoader(eval_dataset_set , batch_size=args.per_device_eval_batch_size, collate_fn=train_data_set.collate_batch, num_workers=4,sampler=val_sampler if val_sampler else None)
    
    
    args.idnum = idnum
    # inint model
    #---------------------------------------------------------------------------------
    model = PreferCodeLlama(args)
    # freeze pretrained model parameters   
    
    
    if train_config.enable_fsdp:
        
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        #model = model.to(local_rank)
        print("1", local_rank, rank)
        model = FSDP(
            model,
            auto_wrap_policy= wrapping_policy,
            cpu_offload=None,
            mixed_precision= mixed_precision_policy,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states= False,
            param_init_fn= None,
        )
        # rank = dist.get_rank()
        # print(f"Start running basic DDP example on rank {rank}.")
        # # create model and move it to GPU with id rank
        # device_id = rank % torch.cuda.device_count()
        # model = model.to(rank)        
        # model = DDP(model, device_ids= [device_id], output_device= device_id, find_unused_parameters=True)
    else:
        model = model.cuda()
    
    if args.freezeLM:
           for name, param in model.module.model.named_parameters():
                param.requires_grad=False
    if not train_config.enable_fsdp or rank == 0:
        print_trainable_parameters(model)
    
    on_gpu = all(param.is_cuda for param in model.parameters())
    if on_gpu:
        print("模型在 GPU 上运行。")
    else:
        print("模型在 CPU 上运行。")
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
            fsdp_config if train_config.enable_fsdp else None,
            local_rank if train_config.enable_fsdp else None,
            rank if train_config.enable_fsdp else None,
        )  
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
if __name__ == "__main__":
    main()