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
import torch
import torch.nn as nn
import torch.nn.functional as F
class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        """
        输入:
            features: 输入样本的特征，尺寸为 [batch_size, hidden_dim].
            labels: 每个样本的ground truth标签，尺寸是[batch_size].
            mask: 用于对比学习的mask，尺寸为 [batch_size, batch_size], 如果样本i和j属于同一个label，那么mask_{i,j}=1 
        输出:
            loss值
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        # 关于labels参数
        if labels is not None and mask is not None:  # labels和mask不能同时定义值，因为如果有label，那么mask是需要根据Label得到的
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None: # 如果没有labels，也没有mask，就是无监督学习，mask是对角线为1的矩阵，表示(i,i)属于同一类
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None: # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        '''
        logits是anchor_dot_contrast减去每一行的最大值得到的最终相似度
        示例: logits: torch.size([4,4])
        logits:
            tensor([[ 0.0000, -0.0471, -0.3352, -0.2156],
                    [-1.2576,  0.0000, -0.3367, -0.0725],
                    [-1.3500, -0.1409, -0.1420,  0.0000],
                    [-1.4312, -0.0776, -0.2009,  0.0000]])       
        '''
        # 构建mask 
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size)     
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        '''
        但是对于计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        # 第ind行第ind位置填充为0
        得到logits_mask:
            tensor([[0., 1., 1., 1.],
                    [1., 0., 1., 1.],
                    [1., 1., 0., 1.],
                    [1., 1., 1., 0.]])
        positives_mask:
        tensor([[0., 0., 1., 1.],
                [0., 0., 0., 0.],
                [1., 0., 0., 1.],
                [1., 0., 1., 0.]])
        negatives_mask:
        tensor([[0., 1., 0., 0.],
                [1., 0., 1., 1.],
                [0., 1., 0., 0.],
                [0., 1., 0., 0.]])
        '''        
        num_positives_per_row  = torch.sum(positives_mask , axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]       
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)  
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
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

    
class PreferAugCodeLlama(nn.Module):
    
    def __init__(self, config):
        super(PreferAugCodeLlama, self).__init__()
        self.model = LlamaForCausalLM.from_pretrained(config.model_name_or_path) 
        self.model.resize_token_embeddings(self.model.model.embed_tokens.weight.size(0) + 8)
        
                    
        self.user_len = 1
        self.emsize = self.model.model.embed_tokens.weight.size(1)  #
        self.user_embeddings = nn.Embedding(config.idnum * self.user_len, self.emsize)
        print(">>> config.idnum = " + str(config.idnum))
        print(">>> self.user_len = " + str(self.user_len))
        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        
        self.mlp = MLP(self.emsize,self.emsize,int(self.emsize/2), 0)
        self.style_head = nn.Linear(self.emsize, config.vocab_size + 8, bias=False)
        self.story_loss_fn = nn.NLLLoss(ignore_index=-100)
        self.para = torch.nn.Parameter(torch.tensor([-2.0]), requires_grad=True)
        self.para.data = self.para.data
        self.enable_contrast = config.enable_contrast
        #print(self.para.item())
    
      
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
            Style_logits = self.style_head(Style_time_hidden_states)
            P_Style = torch.nn.functional.softmax(Style_logits, dim=-1) 
            val =  torch.sigmoid(self.para)
            P_final = P_base*(1-val) + val*P_Style
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
            
            #loss = output.loss
            #hidden_states = outputs[0]
            #logits = self.lm_head(hidden_states)
            base_logits = output.logits
            P_base = torch.nn.functional.softmax(base_logits, dim=-1) 
            hidden_states = output.hidden_states
            #print(base_logits.shape)
            Style_time_hidden_states = self.mlp(hidden_states[-1])
            #print(Style_time_hidden_states.shape)
            #input()
            Style_logits = self.style_head(Style_time_hidden_states)
            #print(Style_logits.shape)
            #input()
            #大小不一致
            P_Style = torch.nn.functional.softmax(Style_logits, dim=-1) 
            val = torch.sigmoid(self.para)
            P_final = P_base*(1-val) + val*P_Style
            
             
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
                Style_hidden_states = torch.mean(Style_time_hidden_states,dim=1)
                u_prompt_batch = gather_all_with_local_grad(u_prompt)
                Style_hidden_states_batch = gather_all_with_local_grad(Style_hidden_states)
                user_id_batch = gather_all_with_local_grad(user_id)
                
                
            return {"loss":story_loss, "para":val}
        