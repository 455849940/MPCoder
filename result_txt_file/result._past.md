## 10个学生 baseline
----------------------------------
bleu-4 = 0.3852491130407462
rouge_1/f_score  0.3209
rouge_1/r_score  0.3656
rouge_1/p_score  0.3097
rouge_2/f_score  0.1671
rouge_2/r_score  0.1869
rouge_2/p_score  0.1745
rouge_l/f_score  0.2647
rouge_l/r_score  0.2773
rouge_l/p_score  0.3734
----------------------------------
0.7121
0.8816
## 10个学生 PreferCodeLlama
----------------------------------
bleu-4 = 0.5074023000348004
rouge_1/f_score  0.3365
rouge_1/r_score  0.3899
rouge_1/p_score  0.3160
rouge_2/f_score  0.1770
rouge_2/r_score  0.2056
rouge_2/p_score  0.1737
rouge_l/f_score  0.2860
rouge_l/r_score  0.3722
rouge_l/p_score  0.3207
-----------------------------------
10轮全冻结 仍然未收敛
bleu-4 = 0.3505796426234685
rouge_1/f_score  0.3094
rouge_1/r_score  0.3423
rouge_1/p_score  0.3197
rouge_2/f_score  0.1549
rouge_2/r_score  0.1714
rouge_2/p_score  0.1696
rouge_l/f_score  0.2356
rouge_l/r_score  0.2630
rouge_l/p_score  0.3797
------------------------------------
40轮全冻结 仍然未收敛
bleu-4 = 0.41295310507890015
rouge_1/f_score  0.3256
rouge_1/r_score  0.3410
rouge_1/p_score  0.3460
rouge_2/f_score  0.1691
rouge_2/r_score  0.1754
rouge_2/p_score  0.1924
rouge_l/f_score  0.2421
rouge_l/r_score  0.2588
rouge_l/p_score  0.3826
----------------------------------
module_classes ={
            LlamaDecoderLayer,
            LlamaRMSNorm,
            nn.Embedding,
            nn.Linear
        }
不冻结llama的Embedding
20轮收敛
id =A
bleu-4 = 0.4811838650902727
rouge_1/f_score  0.3324
rouge_1/r_score  0.3907
rouge_1/p_score  0.3261
rouge_2/f_score  0.1714
rouge_2/r_score  0.2151
rouge_2/p_score  0.1752
rouge_l/f_score  0.2531
rouge_l/r_score  0.3633
rouge_l/p_score  0.3075
-----------------------------------
id = B
4轮收敛
不冻结llama的 nn.Linear, LlamaRMSNorm
bleu-4 = 0.4980836481417678
rouge_1/f_score  0.3265
rouge_1/r_score  0.3677
rouge_1/p_score  0.3212
rouge_2/f_score  0.1550
rouge_2/r_score  0.1738
rouge_2/p_score  0.1606
rouge_l/f_score  0.2393
rouge_l/r_score  0.3205
rouge_l/p_score  0.2792
--------------------------------------

--------------------------------------
## 50个学生 PreferCodeLlama
--------------------------------------
[不冻结llama的Embedding]
model_E.model 未收敛6轮

--------------------------------------
## 100个学生 PreferCodeLlama
[不冻结llama的nn.Linear, LlamaRMSNorm]

--------------------------------------





--------------------------------------