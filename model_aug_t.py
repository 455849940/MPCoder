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
import torch.distributed as dist

import torch.nn.functional as F
class ContrastLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(ContrastLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        
    def forward(self, features_q,features_p, labels=None, mask=None):
        """
        输入:
            features_q: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            features_p: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1 
        输出:
            loss值
        """
        
        features_q = F.normalize(features_q, p=2, dim=1)
        batch_size = features_q.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None: # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).cuda()
        elif labels is not None: # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features_q')
            mask = torch.eq(labels, labels.T).float().cuda()
 
        else:
            mask = mask.float().cuda()
        
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features_q, features_p.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
       
        # 构建mask 
        logits_mask = torch.ones_like(mask) #- torch.eye(batch_size)     
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        
        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 正样本的个数  [2 0 2 2]       
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)  
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            print(logits)
            print(denominator)
            print(exp_logits)
            print(positives_mask)
            raise ValueError("Log_prob has nan!")
        
        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]

        #计算正样本平均的log-likelihood
        
        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
    
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout=0):
        super().__init__()
   
        #self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            #nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        #x = self.dropout(x)
        #x = self.linear(x)
        #x = self.activation(x)
        x = self.layers(x)
        #x = self.activation(x) #新加激活层
        return x
    
def gather_all_with_local_grad(tensor, dim=0):
    local_rank = torch.distributed.get_rank()

    with torch.no_grad():
        all_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(all_tensors, tensor)
    all_tensors[local_rank] = tensor

    return torch.cat(all_tensors, dim=dim)
    
class PreferAugTCodeLlama(nn.Module):
    
    def __init__(self, config):
        super(PreferAugTCodeLlama, self).__init__()
        self.model = LlamaForCausalLM.from_pretrained(config.model_name_or_path) 
        self.model.resize_token_embeddings(self.model.model.embed_tokens.weight.size(0) + 8)
        
                    
        self.user_len = 5
        self.emsize = self.model.model.embed_tokens.weight.size(1)  #
        self.user_embeddings = nn.Embedding(config.idnum * self.user_len, self.emsize)
        print(">>> config.idnum = " + str(config.idnum))
        print(">>> self.user_len = " + str(self.user_len))
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        SCALE = 8
        self.mlp = MLP(self.emsize,self.emsize,int(self.emsize//SCALE), 0)
        #self.style_head = nn.Linear(self.emsize, config.vocab_size + 8, bias=True)
        self.story_loss_fn = nn.NLLLoss(ignore_index=-100)
        
        
        self.keyword_W = nn.Linear(self.emsize, self.model.config.vocab_size)
        self.p_linear1 = nn.Linear(self.model.config.vocab_size, self.emsize//SCALE)
        self.p_linear2 = nn.Linear(self.emsize//SCALE, self.model.config.vocab_size)
        self.p_linear3 = nn.Linear(self.emsize, self.emsize//SCALE)
        #print(self.para.item())
        self.enable_contrast = config.enable_contrast
        self.ContrastLoss = ContrastLoss(temperature=0.5)
        #print(self.para.item())
        self.alpha = config.alpha
      
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
            modeling_outputs = self.model.forward(inputs_embeds=src_emd,past_key_values = past_key_values, output_hidden_states = True)
            
            base_logits = modeling_outputs.logits
            P_base = torch.nn.functional.softmax(base_logits, dim=-1) 
            hidden_states = modeling_outputs.hidden_states
            Style_time_hidden_states = self.mlp(hidden_states[-1])
            average_sequence_tensors = torch.mean(Style_time_hidden_states, dim=1)
            Style_logits = self.keyword_W(Style_time_hidden_states)
            P_Style = torch.nn.functional.softmax(Style_logits, dim=-1)
            #seq_len = base_logits.size(1)
            #P_Style = keyword.unsqueeze(1).repeat(1, seq_len, 1)
            temp = torch.relu(self.p_linear1(P_Style))    # [batch_size, seq_len, bart_last_hidden/SCALE]
            p = self.p_linear2((temp + self.p_linear3(hidden_states[-1])) / 2) # [batch_size, seq_len, vocab_size]
            p = torch.sigmoid(p)
            P_final = p * P_Style + (1-p) * P_base     
            P_final = P_final / P_final.sum(dim=-1, keepdim=True)
            P_final = P_final.log()
            
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
            
            
            base_logits = output.logits
            P_base = torch.nn.functional.softmax(base_logits, dim=-1) 
            hidden_states = output.hidden_states
            #average_sequence_tensors = torch.mean(u_emd, dim=1)
            Style_time_hidden_states = self.mlp(hidden_states[-1])
            average_sequence_tensors = torch.mean(Style_time_hidden_states, dim=1)
            Style_logits = self.keyword_W(Style_time_hidden_states)
            
            P_Style = torch.nn.functional.softmax(Style_logits, dim=-1)
            #seq_len = base_logits.size(1)
            
            #P_Style = keyword.unsqueeze(1).repeat(1, seq_len, 1)
            temp = torch.relu(self.p_linear1(P_Style))    # [batch_size, seq_len, bart_last_hidden/SCALE]
            p = self.p_linear2((temp + self.p_linear3(hidden_states[-1])) / 2) # [batch_size, seq_len, vocab_size]
            p = torch.sigmoid(p)
            P_final = p * P_Style + (1-p) * P_base     
            P_final = P_final / P_final.sum(dim=-1, keepdim=True)
            P_final = P_final.log()
            
            
            # Flatten the tokens
            shift_P_final = P_final[..., :-1, :].contiguous()
            shift_labels = newlabels[..., 1:].contiguous()
            
            
            shift_P_final = shift_P_final.view(-1, self.model.config.vocab_size)
            
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_P_final.device)
            
            story_loss = self.story_loss_fn(shift_P_final,shift_labels)
            if self.enable_contrast:
                u_prompt = torch.mean(u_emd,dim=1)
                #Style_hidden_states = torch.mean(average_sequence_tensors,dim=1)
                u_prompt_batch = gather_all_with_local_grad(u_prompt)
                Style_hidden_states_batch = gather_all_with_local_grad(average_sequence_tensors)
                user_id_batch = gather_all_with_local_grad(user_id)
                contrast_loss = self.ContrastLoss(u_prompt_batch,Style_hidden_states_batch,labels = user_id_batch)
                story_loss =  story_loss +  self.alpha * contrast_loss
            
            return {"loss":story_loss}
        