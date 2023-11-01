from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from transformers import (
    LlamaPreTrainedModel,
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig
)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0):
        super().__init__()
   
        #self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.dropout(x)
        #x = self.linear(x)
        #x = self.activation(x)
        x = self.layers(x)
        return x

    
class PreferAugCodeLlama(nn.Module):
    
    def __init__(self, config):
        super(PreferAugCodeLlama, self).__init__()
        self.model = LlamaForCausalLM.from_pretrained(config.model_name_or_path) 
        self.model.resize_token_embeddings(self.model.model.embed_tokens.weight.size(0) + 8)
        
                    
        self.user_len = 2
        self.emsize = self.model.model.embed_tokens.weight.size(1)  #
        self.user_embeddings = nn.Embedding(config.idnum * self.user_len, self.emsize)
        print(">>> config.idnum = " + str(config.idnum))
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        
        self.mlp = MLP(self.emsize,self.emsize,int(self.emsize/2), 0)
        self.style_head = nn.Linear(self.emsize, config.vocab_size + 8, bias=True)
        self.story_loss_fn = nn.NLLLoss(ignore_index=-100)
        
    def forward(self, user_id, input_ids, attention_mask, past_key_values = None):
        #print("now")
        ignore_index = -100
        text = input_ids
        mask = attention_mask
        batch_size = user_id.size(0)
        user_id = user_id * self.user_len
        user_id_sql = torch.Tensor(user_id.unsqueeze(1).repeat(1, self.user_len).view(-1, self.user_len)).cuda()
        add_id_sql = torch.arange(0, self.user_len).cuda()
        id_sql = user_id_sql + add_id_sql
        u_emd = self.user_embeddings(id_sql)
        #print("1", user_id.device, user_id_sql.device, add_id_sql.device, id_sql.device)
        w_emd = self.model.model.embed_tokens(input_ids)  # (batch_size, tgt_len, emsize)
        if past_key_values is None:
            src_emd = torch.cat([u_emd, w_emd], 1)  # (batch_size, total_len, emsize)
        else:
            src_emd = w_emd
        if mask is None:
            # auto-regressive generation
            modeling_outputs = self.model.forward(inputs_embeds=src_emd,past_key_values = past_key_values)
            base_logits = modeling_outputs.logits
            P_base = torch.nn.functional.softmax(base_logits, dim=-1) 
            hidden_states = output.hidden_states
            
            Style_time_hidden_states = self.mlp(hidden_states)
            Style_logits = self.style_head(Style_time_hidden_states)
            P_Style = torch.nn.functional.softmax(Style_logits, dim=-1) 
            P_final = P_base + P_Style
            P_final = P_final / P_final.sum(dim=-1, keepdim=True)
            
            return modeling_outputs, P_final
            #新推理提供以上两个值，需要下一token的词表分布向量
        else:
            pad_left = torch.ones((batch_size, self.user_len)).cuda()
            pad_mask = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)
            
            # prediction for training
            pred_left = torch.full((batch_size, self.user_len), ignore_index, dtype=torch.int64).cuda()  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).cuda())  # replace <pad> with ignore_index
            newlabels = torch.cat([pred_left, pred_right], 1)
            
            output = self.model(
                #input_ids=input_ids,
                inputs_embeds = src_emd,
                attention_mask = pad_mask,
                labels= newlabels,
                return_dict = True,
                output_hidden_states = True
            )
            
            #loss = output.loss
            #hidden_states = outputs[0]
            #logits = self.lm_head(hidden_states)
            base_logits = output.logits
            P_base = torch.nn.functional.softmax(base_logits, dim=-1) 
            hidden_states = output.hidden_states
            #print(base_logits.shape)
            Style_time_hidden_states = self.mlp(hidden_states[-1])
            Style_logits = self.style_head(Style_time_hidden_states)
            #print(Style_logits.shape)
            #input()
            #大小不一致
            P_Style = torch.nn.functional.softmax(Style_logits, dim=-1) 
            
            P_final = P_base + P_Style
            P_final = P_final / P_final.sum(dim=-1, keepdim=True)
            P_final = P_final.log()
            
            # Flatten the tokens
            shift_P_final = P_final[..., :-1, :].contiguous()
            shift_labels = newlabels[..., 1:].contiguous()
            
            
            shift_P_final = shift_P_final.view(-1, self.model.config.vocab_size)
            
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_P_final.device)
            
            story_loss = self.story_loss_fn(shift_P_final,shift_labels)
             
            
            return story_loss
        