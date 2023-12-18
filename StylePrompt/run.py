from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import json
import os
from feature_arguments import train_config
import torch
import transformers 
from transformers import LlamaTokenizer
from torch.utils.data import DistributedSampler
from load_feature_data import FeatureProcessClass
from load_user_data import processClass
from FeatureModel import PreferFeatureCodeLlama
from feature_train_utils import load_model_checkpoint,F_train
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from user_train_utils import train, clear_gpu_cache, setup_environ_flags, get_policies
import torch.distributed as dist
from configs.fsdp import fsdp_config
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload
)

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
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    if torch.distributed.is_initialized():
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
    # load feature data
    #---------------------------------------------------------------------------------  
    if args.do_train_first:
        Feature_proc = FeatureProcessClass(tokenizer)
        train_Feature_data_set = Feature_proc.get_dataset(args,tokenizer,  choose = "train",is_test = False,rank = rank)
        dev_Feature_data_set = Feature_proc.get_dataset(args,tokenizer,  choose = "dev", is_test = False,rank = rank)
        train_Feature_sampler = None
        dev_Feature_sampler = None
        if train_config.enable_fsdp:
            train_Feature_sampler = DistributedSampler(
                train_Feature_data_set,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=True, #shuffle
            )
            if train_config.do_eval:
                dev_Feature_sampler = DistributedSampler(
                    dev_Feature_data_set,
                    rank=dist.get_rank(),
                    num_replicas=dist.get_world_size(),
                )
        
        
        F_train_dataloader = DataLoader(train_Feature_data_set , batch_size=args.per_device_feature_train_batch_size, collate_fn=train_Feature_data_set.collate_batch, num_workers=4,sampler=train_Feature_sampler)
        F_eval_dataloader = DataLoader(dev_Feature_data_set , batch_size=args.per_device_feature_dev_batch_size, collate_fn=dev_Feature_data_set.collate_batch, num_workers=4,sampler=dev_Feature_sampler)
    
    # load data
    #---------------------------------------------------------------------------------  
    proc = processClass(args.user_style_data_path)
    train_data_set = proc.get_train_dataset(args,tokenizer,  is_test = False,rank = rank)
    idnum = proc.idnum
    if not train_config.enable_fsdp or rank == 0:
        print(idnum)
        print("--------")
    eval_dataset_set = proc.get_eval_datasets(args,tokenizer,  is_test = False,rank = rank)
    idnum = 50
    args.idnum = idnum 
    #--------------------------------------------------------------------------------- 
    

    model = PreferFeatureCodeLlama(args)
    

    
    #print_trainable_parameters(model)
    if args.do_train_first:    
        if args.continue_train == True:
            model = load_model_checkpoint(model, 0, args.output_dir)
        
        
        
        if args.freezeLM:
             for name, param in model.named_parameters():
                if "Style_embeddings" in name : continue
                param.requires_grad=False
                
                    
        #fsdp_config.choose_model_name = args.choose_model_name
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        print("1", local_rank, rank)
        model = FSDP(
            model,
            auto_wrap_policy= wrapping_policy,
            cpu_offload=None, #CPUOffload(offload_params=True)
            mixed_precision= mixed_precision_policy,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states= False,
            param_init_fn= None,
        )
        
        if not train_config.enable_fsdp or rank == 0:
            print_trainable_parameters(model)
        
        on_gpu = all(param.is_cuda for param in model.parameters())
        if rank == 0:
            if on_gpu:
                print("模型在 GPU 上运行。")
            else:
                print("模型在 CPU 上运行。")
    # build model and trainer
    #---------------------------------------------------------------------------------

    
        
    if args.do_train_first:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        results = F_train(
            model,
            F_train_dataloader,
            F_eval_dataloader,
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
    
    ## second stage
    ##---------------------------------------------------------------------------------
    
    
    if args.do_train_second:
        print("stage 2")
        #inint model
        #---------------------------------------------------------------------------------
        if args.do_train_first == False:
            if args.continue_train == True:
                model = load_model_checkpoint(model, 0, args.output_dir2)
            else:
                model = load_model_checkpoint(model, 0, args.output_dir)
        
        
        
        model.set_forwardChoose(1)
        if args.freezeLM:
            for name, param in model.named_parameters():
                if "Style_embeddings" in name : 
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    
            for name, param in model.model.named_parameters():
                if "lm_head" in name or "model.norm" in name: 
                    param.requires_grad =True
                else:
                    param.requires_grad =False
                    
        if args.do_train_first == False:
            
            #fsdp_config.choose_model_name = args.choose_model_name
            mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
            print("2", local_rank, rank)
            model = FSDP(
                model,
                auto_wrap_policy= wrapping_policy,
                cpu_offload= CPUOffload(offload_params=True),
                mixed_precision= mixed_precision_policy,
                sharding_strategy=fsdp_config.sharding_strategy,
                device_id=torch.cuda.current_device(),
                limit_all_gathers=True,
                sync_module_states= False,
                param_init_fn= None,
            )   
        
                     
        if not train_config.enable_fsdp or rank == 0:
                print_trainable_parameters(model)
        on_gpu = all(param.is_cuda for param in model.parameters())
        if rank == 0:
            if on_gpu:
                print("模型在 GPU 上运行。")
            else:
                print("模型在 CPU 上运行。")
        ## create dataloader
        #---------------------------------------------------------------------------------                
        train_sampler = None
        val_sampler = None
        if train_config.enable_fsdp:
            train_sampler = DistributedSampler(
                train_data_set,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=True, #shuffle
            )
            if train_config.do_eval:
                val_sampler = DistributedSampler(
                    eval_dataset_set,
                    rank=dist.get_rank(),
                    num_replicas=dist.get_world_size(),
                )
        
        train_dataloader = DataLoader(train_data_set , batch_size=args.per_device_train_batch_size, collate_fn=train_data_set.collate_batch, num_workers=4,sampler=train_sampler if train_sampler else None)
        eval_dataloader = DataLoader(eval_dataset_set , batch_size=args.per_device_eval_batch_size, collate_fn=train_data_set.collate_batch, num_workers=4,sampler=val_sampler if val_sampler else None)
        
        
        optimizer2 = optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler2 = StepLR(optimizer2, step_size=1, gamma=args.gamma)
        #train
        #---------------------------------------------------------------------------------
        results = train(
            model,
            train_dataloader,
            eval_dataloader,
            tokenizer,
            optimizer2,
            scheduler2,
            args.gradient_accumulation_steps,
            args,
            fsdp_config if train_config.enable_fsdp else None,
            local_rank if train_config.enable_fsdp else None,
            rank if train_config.enable_fsdp else None,
        )  
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
        
    
if __name__ == "__main__":
    main()