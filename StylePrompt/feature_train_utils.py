# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging

from torch import nn
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
from memory_utils import MemoryTrace
from llama_recipes.policies import fpSixteen,bfSixteen_mixed
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig
)
import functools

from transformers.models.llama.modeling_llama import LlamaDecoderLayer,LlamaRMSNorm,LlamaForCausalLM
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    _module_wrap_policy,
)


fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
def save_model_checkpoint(
    model,
    rank,
    output_dir,
    epoch=1,
):
    """saving model via rank0 cpu streaming and full_state_dict"""
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fullstate_save_policy
    ):
        cpu_state = model.state_dict()
        print(f"saving process: rank {rank}  done w model state_dict")
    if rank == 0:
        #output_dir = os.path.join(cfg.output_dir, f'model_{epoch}.pt')
        output_dir = os.path.join(output_dir, f'model.pt')
        torch.save(cpu_state, output_dir)
        print(f"model checkpoint saved for epoch {epoch} at {output_dir}\n")
      


def load_model_checkpoint(model, rank,output_dir):
    if rank != 0:
        return
    
    model_checkpoint_dir = os.path.join(output_dir, f'model.pt')
    print(f"start model checkpoint loaded in {model_checkpoint_dir}")
    model_checkpoint = torch.load(model_checkpoint_dir)
    model.load_state_dict(model_checkpoint)
    print(f"model checkpoint loaded in {model_checkpoint_dir}")
    return model


def F_train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None):
    """
    Trains the model on the given dataloader
    """


    
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    train_prep = []
    train_loss = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    early_stop_patience = 1  # 当连续1次验证集性能没有提升时停止训练
    early_stop_counter = 0
    quit_flag = False
    if train_config.continue_train == True :
        eval_ppl, eval_epoch_loss = F_evaluation(model, train_config, eval_dataloader, tokenizer, local_rank)
        best_val_loss = eval_epoch_loss
        if rank==0:
            print(f"continue_trian now best_val_loss ininit:{best_val_loss}")
    
    is_main_process = dist.get_rank() == 0             
    for epoch in range(int(train_config.num_feature_train_epochs)):
        epoch_start_time = time.perf_counter()
        if quit_flag == True: break
        
        with MemoryTrace() as memtrace:
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            eval_interval = total_length//1  # 设置每隔多少个训练批次进行一次验证
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True, leave=is_main_process)    
            for step, batch in enumerate(train_dataloader):
                if quit_flag == True: break 
                
                if train_config.enable_fsdp:
                    input_ids = batch['input_ids'].to(local_rank)
                    attention_mask = batch['attention_mask'].to(local_rank)
                    select_mask = batch['select_mask'].to(local_rank)
                else:    
                    input_ids = batch['input_ids'].cuda()
                    attention_mask = batch['attention_mask'].cuda()
                    select_mask = batch['select_mask'].cuda()
                
                result = model(user_id = None,input_ids = input_ids,attention_mask = attention_mask,select_mask = select_mask, past_key_values = None)

                loss = result["loss"]
                loss = loss / gradient_accumulation_steps
                total_loss += loss.detach().float()
  
                # regular backpropagation when fp16 is not used
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    optimizer.zero_grad()
                    pbar.update(1)
                    
                if (step + 1) % eval_interval == 0:
                    if train_config.do_eval:
                        eval_ppl, eval_epoch_loss = F_evaluation(model, train_config, eval_dataloader, tokenizer, local_rank)
                        checkpoint_start_time = time.perf_counter()
                        if train_config.save_model and eval_epoch_loss < best_val_loss:
                            early_stop_counter = 0
                            if train_config.enable_fsdp:
                                dist.barrier()
                            save_model_checkpoint(model,rank,train_config.output_dir, epoch)
                            if rank==0:
                                print(" Saving the model checkpoints." + f"{epoch+1}")
                                print("=====================================================")                     
                            if train_config.enable_fsdp:
                                dist.barrier()
                        else:
                            early_stop_counter += 1
                            if early_stop_counter == early_stop_patience:
                                quit_flag = True
                                
                        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                        checkpoint_times.append(checkpoint_end_time)
                        if eval_epoch_loss < best_val_loss:
                            best_val_loss = eval_epoch_loss
                            if train_config.enable_fsdp:
                                if rank==0:
                                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                            else:
                                print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                                
                    model.train()
                    
                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_feature_train_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
                torch.cuda.empty_cache()
            pbar.close()
            
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)    
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
           
            
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)
        
        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        
        # Update the learning rate as needed
        lr_scheduler.step()
          
        
        if rank==0:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
            
            
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    return results

def F_evaluation(model,train_config, eval_dataloader, tokenizer, local_rank):
    """
    Evaluates the model on the given dataloader
    
    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions
    
    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])  
    model.eval()
  
    eval_loss = 0.0  # Initialize evaluation loss
    
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            if train_config.enable_fsdp:
                input_ids = batch['input_ids'].to(local_rank)
                attention_mask = batch['attention_mask'].to(local_rank)
                select_mask = batch['select_mask'].to(local_rank)
            else:
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                select_mask = batch['select_mask'].cuda()
                                
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                result = model(user_id = None,input_ids = input_ids,attention_mask = attention_mask,select_mask = select_mask, past_key_values = None)
                loss = result["loss"]
                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            #preds = torch.argmax(logits, -1)
            #eval_preds.extend(
            #    tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            #)
    
    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    
    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)
    
    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")
        
    return eval_ppl, eval_epoch_loss



                    

                    


