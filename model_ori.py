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
class PromptModule(nn.Module):
    def __init__(self, nid, mbed_tokens):
        self.user_len = 20
        self.mbed_tokens = mbed_tokens
        self.emsize = self.mbed_tokens.weight.size(1)  # 768
        self.user_embeddings = nn.Embedding(nid, self.emsize)
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, user_id , text, mask, ignore_index=-100):
        batch_size = user_id.size(0)
        # embeddings
        # 使用索引操作获取向量
        indices = torch.arange(user_id*self.user_len, (user_id+1)*self.user_len)  # 创建索引范围
        u_emd = self.user_embeddings(indices)  # 使用索引获取向量
        w_emd = self.mbed_tokens(text)  # (batch_size, tgt_len, emsize)

        src_emd = torch.cat([u_emd.unsqueeze(1), w_emd], 1)  # (batch_size, total_len, emsize)

        # training
        # input padding
        pad_left = torch.ones((batch_size, self.user_len), dtype=torch.int64).cuda()
        pad_mask = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

        # prediction for training
        pred_left = torch.full((batch_size, self.user_len), ignore_index, dtype=torch.int64).cuda()  # (batch_size, src_len)
        pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).cuda())  # replace <pad> with ignore_index
        labels = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

        return pad_mask, src_emd, labels
        
class PreferCodeLlama(nn.Module):
    
    def __init__(self, config):
        super(PreferCodeLlama, self).__init__()
        self.model = LlamaForCausalLM.from_pretrained(config.model_name_or_path) 
        # freeze pretrained model parameters        
        if config.freezeLM:
           for name, param in self.model.named_parameters():
                #print(name)
                param.requires_grad=False
                
        self.user_len = 20
        self.emsize = self.model.model.embed_tokens.weight.size(1)  #
        self.user_embeddings = nn.Embedding(config.idnum * self.user_len, self.emsize)
        print(">>> config.idnum = " + str(config.idnum))
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, user_id, input_ids, attention_mask, labels ):
        ignore_index = -100
        text = input_ids
        mask = attention_mask
        batch_size = user_id.size(0)
        user_id = user_id * self.user_len
        user_id_sql = torch.Tensor(user_id.unsqueeze(1).repeat(1, self.user_len).view(-1, self.user_len)).cuda()
        add_id_sql = torch.arange(0, self.user_len).cuda()
        id_sql = user_id_sql + add_id_sql
        u_emd = self.user_embeddings(id_sql)

        w_emd = self.model.model.embed_tokens(input_ids)  # (batch_size, tgt_len, emsize)
        src_emd = torch.cat([u_emd, w_emd], 1)  # (batch_size, total_len, emsize)
        
        
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
        logits = output.logits
        hidden_states = output.hidden_states
        predictions = logits.argmax(dim=-1)
        
        return loss, logits, hidden_states, predictions
        