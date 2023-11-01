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

        
class PreferCodeLlama(nn.Module):
    
    def __init__(self, config):
        super(PreferCodeLlama, self).__init__()
        self.model = LlamaForCausalLM.from_pretrained(config.model_name_or_path) 
        self.model.resize_token_embeddings(self.model.model.embed_tokens.weight.size(0) + 8)
                     
        self.user_len = 5
        self.emsize = self.model.model.embed_tokens.weight.size(1)  #
        self.user_embeddings = nn.Embedding(config.idnum * self.user_len, self.emsize)
        print(">>> config.idnum = " + str(config.idnum))
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        
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
            return self.model.forward(inputs_embeds=src_emd,past_key_values = past_key_values)
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
                return_dict = True
            )
            
            loss = output.loss
            #logits = output.logits
            #hidden_states = output.hidden_states
            #predictions = logits.argmax(dim=-1)
            
            return loss
        