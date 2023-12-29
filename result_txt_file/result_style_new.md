## Java测评 30学生版本
-------------------------------
## baseline 
bleu-4 = 0.49148893105257696
rouge_1/f_score  0.3838
rouge_1/r_score  0.4056
rouge_1/p_score  0.3960
rouge_2/f_score  0.2660
rouge_2/r_score  0.2797
rouge_2/p_score  0.2805
total_eval_val:0.7833326349285525
-------------------------------
## style_model Linear
bleu-4 = 0.5109717816791142
rouge_1/f_score  0.3936
rouge_1/r_score  0.4098
rouge_1/p_score  0.4147
rouge_2/f_score  0.2737
rouge_2/r_score  0.2869
rouge_2/p_score  0.2953
fail_convert_java_cont:2
total_eval_val:0.8025151258737928
--------------------------------
## style_model Linear batch 4
bleu-4 = 0.5010623247298589
rouge_1/f_score  0.4079
rouge_1/r_score  0.4262
rouge_1/p_score  0.4256
rouge_2/f_score  0.2884
rouge_2/r_score  0.3030
rouge_2/p_score  0.3077
rouge_l/f_score  0.4291
rouge_l/r_score  0.5301
rouge_l/p_score  0.5286
--------------------------------
## 30学生 style_aug Linear lr 1e-4 2轮收敛
bleu-4 = 0.5002164138169433
rouge_1/f_score  0.3913
rouge_1/r_score  0.4087
rouge_1/p_score  0.4107
rouge_2/f_score  0.2668
rouge_2/r_score  0.2801
rouge_2/p_score  0.2870
fail_convert_java_cont:5
total_eval_val:0.8117713720757292
--------------------------------
## 30学生 style_augT Linear lr 1e-4 2轮收敛
bleu-4 = 0.49859913387417
rouge_1/f_score  0.4103
rouge_1/r_score  0.4250
rouge_1/p_score  0.4305
rouge_2/f_score  0.2911
rouge_2/r_score  0.3025
rouge_2/p_score  0.3119
fail_convert_java_cont:3
total_eval_val:0.7972220188806449
--------------------------------





## Java测评 50学生版本
-------------------------------
## baseline 
bleu-4 = 0.4071491550170681
rouge_1/f_score  0.3497
rouge_1/r_score  0.3900
rouge_1/p_score  0.3493
rouge_2/f_score  0.2147
rouge_2/r_score  0.2388
rouge_2/p_score  0.2203
fail_convert_java_cont:4
total_eval_val:0.7668071179296584
-------------------------------
## style_model Linear
bleu-4 = 0.5430623665672782
rouge_1/f_score  0.4066
rouge_1/r_score  0.4223
rouge_1/p_score  0.4161
rouge_2/f_score  0.2727
rouge_2/r_score  0.2848
rouge_2/p_score  0.2838
fail_convert_java_cont:3
total_eval_val:0.8364788266648411
--------------------------------
## style_model Linear batch 4
bleu-4 = 0.5536147929598885
rouge_1/f_score  0.4210
rouge_1/r_score  0.4423
rouge_1/p_score  0.4245
rouge_2/f_score  0.2838
rouge_2/r_score  0.2992
rouge_2/p_score  0.2917
fail_convert_java_cont:1
total_eval_val:0.8288764037127315
--------------------------------
## 50学生 style_aug Linear lr 1e-4 2轮收敛
bleu-4 = 0.5634265097031185
rouge_1/f_score  0.4236
rouge_1/r_score  0.4347
rouge_1/p_score  0.4335
rouge_2/f_score  0.2862
rouge_2/r_score  0.2944
rouge_2/p_score  0.2977
fail_convert_java_cont:0
total_eval_val:0.8161682551451899
--------------------------------
## (1-p) + p版本
bleu-4 = 0.5626287273179416
rouge_1/f_score  0.4267
rouge_1/r_score  0.4339
rouge_1/p_score  0.4400
rouge_2/f_score  0.2876
rouge_2/r_score  0.2920
rouge_2/p_score  0.3032
fail_convert_java_cont:0
total_eval_val:0.8197765302064117
--------------------------------
## (1-p) + p版本 batch 4
糟糕bleu-4 = 0.3463423820557216
--------------------------------
# 对比学习版 （1-p）+p版本 batch 4
bleu-4 = 0.5416601951943119
rouge_1/f_score  0.4097
rouge_1/r_score  0.4236
rouge_1/p_score  0.4223
rouge_2/f_score  0.2734
rouge_2/r_score  0.2842
rouge_2/p_score  0.2866
fail_convert_java_cont:4
total_eval_val:0.8066795724145219
-------------------------------
## 50学生 style_augT Linear lr 1e-4 2轮收敛
bleu-4 = 0.5389745825782536
rouge_1/f_score  0.4076
rouge_1/r_score  0.4307
rouge_1/p_score  0.4095
rouge_2/f_score  0.2748
rouge_2/r_score  0.2931
rouge_2/p_score  0.2797
fail_convert_java_cont:2
total_eval_val:0.8284462534416421
--------------------------------
## 50学生 style_augT Linear batch4
bleu-4 = 0.5413723450036205
rouge_1/f_score  0.4186
rouge_1/r_score  0.4450
rouge_1/p_score  0.4191
rouge_2/f_score  0.2813
rouge_2/r_score  0.3015
rouge_2/p_score  0.2857
fail_convert_java_cont:1
total_eval_val:0.8237980355120122
----------------------------------
## 50学生 style_augT Linear batch4 对比学习 all_emd
bleu-4 = 0.5407418576855734
rouge_1/f_score  0.4131
rouge_1/r_score  0.4379
rouge_1/p_score  0.4159
rouge_2/f_score  0.2753
rouge_2/r_score  0.2930
rouge_2/p_score  0.2827
fail_convert_java_cont:1
total_eval_val:0.8226505396296262
------------------------------------
## 50学生 style_augT Linear batch4 无对比学习## 50学生 U_emd
bleu-4 = 0.5390843298377094
rouge_1/f_score  0.4136
rouge_1/r_score  0.4397
rouge_1/p_score  0.4164
rouge_2/f_score  0.2778
rouge_2/r_score  0.2980
rouge_2/p_score  0.2839
fail_convert_java_cont:2
total_eval_val:0.8258050358347329
---------------------------------------------------
## 50学生 style_augT Linear batch4 对比学习## 50学生 U_emd
bleu-4 = 0.5447187037167036
rouge_1/f_score  0.4162
rouge_1/r_score  0.4406
rouge_1/p_score  0.4187
rouge_2/f_score  0.2792
rouge_2/r_score  0.2970
rouge_2/p_score  0.2861
fail_convert_java_cont:1
total_eval_val:0.8242020281423962
---------------------------------------------
## 50学生 style_augT Linear batch4 无对比学习## 50学生 all_emd+part_code = ht
bleu-4 = 0.5329884982336164
rouge_1/f_score  0.4145
rouge_1/r_score  0.4388
rouge_1/p_score  0.4188
rouge_2/f_score  0.2782
rouge_2/r_score  0.2967
rouge_2/p_score  0.2859
fail_convert_java_cont:2
total_eval_val:0.8285712036805533
---------------------------------------------
## 50学生 style_augT Linear batch4 对比学习## 50学生 all_emd+part_code = ht
bleu-4 = 0.51171892690499
rouge_1/f_score  0.4037
rouge_1/r_score  0.4438
rouge_1/p_score  0.3919
rouge_2/f_score  0.2672
rouge_2/r_score  0.2966
rouge_2/p_score  0.2635
fail_convert_java_cont:3
total_eval_val:0.8312302281825229
--------------------------------------------
## 50学生 style_augT Linear batch4 无对比学习## 50学生 all_emd+part_code = ht SCALE = 8
bleu-4 = 0.5272732379585912
rouge_1/f_score  0.4089
rouge_1/r_score  0.4363
rouge_1/p_score  0.4112
rouge_2/f_score  0.2744
rouge_2/r_score  0.2954
rouge_2/p_score  0.2801
fail_convert_java_cont:2
total_eval_val:0.8241381557230778
--------------------------------------------
## 50学生 style_augT Linear batch4 对比学习## 50学生 all_emd+part_code = ht SCALE = 8
bleu-4 = 0.5290826030709621
rouge_1/f_score  0.4147
rouge_1/r_score  0.4530
rouge_1/p_score  0.4032
rouge_2/f_score  0.2796
rouge_2/r_score  0.3079
fail_convert_java_cont:2
total_eval_val:0.8418502465638122
--------------------------------------------
## 50学生 style_augT Linear batch4 对比学习0.4## 50学生 all_emd+part_code = ht SCALE = 8
bleu-4 = 0.5407494800709552
rouge_1/f_score  0.4142
rouge_1/r_score  0.4520
rouge_1/p_score  0.4030
rouge_2/f_score  0.2782
rouge_2/r_score  0.3068
rouge_2/p_score  0.2744
fail_convert_java_cont:1
total_eval_val:0.8353757681143936
------------------------------------------
## 50学生 style_augT Linear batch4 对比学习0.45 ## 50学生 all_emd+part_code = ht SCALE = 8
bleu-4 = 0.5272732379585912
rouge_1/f_score  0.4089
rouge_1/r_score  0.4363
rouge_1/p_score  0.4112
rouge_2/f_score  0.2744
rouge_2/r_score  0.2954
rouge_2/p_score  0.2801
fail_convert_java_cont:2
total_eval_val:0.8241381557230778
------------------------------------------
## 50学生 style_augT Linear batch4 对比学习0.6 ## 50学生 all_emd+part_code = ht SCALE = 8
bleu-4 = 0.5194329259681576
rouge_1/f_score  0.4141
rouge_1/r_score  0.4527 
rouge_1/p_score  0.4038 
rouge_2/f_score  0.2768
rouge_2/r_score  0.3051
rouge_2/p_score  0.2747
fail_convert_java_cont:3 
total_eval_val:0.8291548850992686  
-----------------------------------------
## 50学生 style_augT Linear batch4 对比学习0.5 ## 50学生 all_emd+part_code = ht SCALE = 8 tanh
bleu-4 = 0.5259904516823547
rouge_1/f_score  0.4071
rouge_1/r_score  0.4434
rouge_1/p_score  0.3997
rouge_2/f_score  0.2710
rouge_2/r_score  0.2988
rouge_2/p_score  0.2696
fail_convert_java_cont:4
total_eval_val:0.8297782101349616
----------------------------------------
提示词1
baseline
{'pass@1': 0.2926829268292683}  48/164
style_model
{'pass@1': 0.21341463414634146} 35/164
Perfer_model
{'pass@1': 0.19018404} 31/164

提示词2
baseline
{'pass@1': 0.27439024390243905} 45/164
style_model
{'pass@1': 0.2621951219512195} 43/164
Perfer_model
{'pass@1': 0.23170731707317074} 38/164

提示词3
baseline
{'pass@1': 0.3475609756097561}
style_model
{'pass@1': 0.32926829268292684}
Perfer_model
{'pass@1': 0.3719512195121951} 61/164
-----------------------------------------------
## 全量学生 java 
baseline
bleu-4 = 0.5042722272877578
rouge_1/f_score  0.3840
rouge_1/r_score  0.3831
rouge_1/p_score  0.4214
rouge_2/f_score  0.2502
rouge_2/r_score  0.2468
rouge_2/p_score  0.2834
rouge_l/f_score  0.4434
rouge_l/r_score  0.4907
rouge_l/p_score  0.5929
fail_convert_java_cont:129
total_eval_val:0.7755816602171647
-------------------------------------------------
## soft promot 1e4
bleu-4 = 0.5836301902774172
rouge_1/f_score  0.4240
rouge_1/r_score  0.4209
rouge_1/p_score  0.4496
rouge_2/f_score  0.2850
rouge_2/r_score  0.2834
rouge_2/p_score  0.3096
rouge_l/f_score  0.4342
rouge_l/r_score  0.5030
rouge_l/p_score  0.5484
fail_convert_java_cont:82
total_eval_val:0.8430786492488279
-------------------------------------------------
## soft promot 1e5
bleu-4 = 0.5902703282330948
rouge_1/f_score  0.4266
rouge_1/r_score  0.4192
rouge_1/p_score  0.4585
rouge_2/f_score  0.2910
rouge_2/r_score  0.2857
rouge_2/p_score  0.3212
rouge_l/f_score  0.4615
rouge_l/r_score  0.5075
rouge_l/p_score  0.6037
fail_convert_java_cont:70
total_eval_val:0.8054808360506226
-------------------------------------------------
## aug_style_model 1e4
bleu-4 = 0.5699764574658542
rouge_1/f_score  0.4059
rouge_1/r_score  0.4078
rouge_1/p_score  0.4274
rouge_2/f_score  0.2657
rouge_2/r_score  0.2682
rouge_2/p_score  0.2867
rouge_l/f_score  0.4096
rouge_l/r_score  0.4868
rouge_l/p_score  0.5202
fail_convert_java_cont:86
total_eval_val:0.8604555648944369
-------------------------------------------------
## aug_style_model 1e5
bleu-4 = 0.5814168170409827
rouge_1/f_score  0.4198
rouge_1/r_score  0.4176
rouge_1/p_score  0.4473
rouge_2/f_score  0.2851
rouge_2/r_score  0.2837
rouge_2/p_score  0.3117
rouge_l/f_score  0.4554
rouge_l/r_score  0.5074
rouge_l/p_score  0.5928
fail_convert_java_cont:78
total_eval_val:0.8173438220057275
------------------------------------------------

-----------------------
baseline 50
bleu-4 = 0.5511718261990197
rouge_1/f_score  0.4328
rouge_1/r_score  0.3969
rouge_1/p_score  0.4967
rouge_2/f_score  0.2918
rouge_2/r_score  0.2647
rouge_2/p_score  0.3473
rouge_l/f_score  0.5031
rouge_l/r_score  0.5301
rouge_l/p_score  0.6609
total_eval_val:0.7369926822460217
---------------------
style model e-5
bleu-4 = 0.6383676512794505
rouge_1/f_score  0.5100
rouge_1/r_score  0.4919
rouge_1/p_score  0.5505
rouge_2/f_score  0.3708
rouge_2/r_score  0.3552
rouge_2/p_score  0.4109
rouge_l/f_score  0.5371
rouge_l/r_score  0.5943
rouge_l/p_score  0.6477
fail_convert_java_cont:8
total_eval_val:0.770802851696928
-----------------------
augT 50 e-5
bleu-4 = 0.6404101943922246
rouge_1/f_score  0.5118
rouge_1/r_score  0.4917
rouge_1/p_score  0.5524
rouge_2/f_score  0.3725
rouge_2/r_score  0.3557
rouge_2/p_score  0.4127
rouge_l/f_score  0.5416
rouge_l/r_score  0.5921
rouge_l/p_score  0.6617
fail_convert_java_cont:6
total_eval_val:0.7786908365386372
----------------------
new 50 model_all_e4
数据1024版 1e-4 /e-5
bleu-4 = 0.6405012442135231
rouge_1/f_score  0.5120
rouge_1/r_score  0.4907
rouge_1/p_score  0.5547
rouge_2/f_score  0.3759
rouge_2/r_score  0.3586
rouge_2/p_score  0.4174
rouge_l/f_score  0.5380
rouge_l/r_score  0.5882
rouge_l/p_score  0.6534
fail_convert_java_cont:4
total_eval_val:0.7673130767169989
-------------------------
数据2048版 1e-4 batch8/ e-5
bleu-4 = 0.6376077982587522
rouge_1/f_score  0.5158
rouge_1/r_score  0.4979
rouge_1/p_score  0.5566
rouge_2/f_score  0.3797
rouge_2/r_score  0.3653
rouge_2/p_score  0.4199
rouge_l/f_score  0.5361
rouge_l/r_score  0.5847
rouge_l/p_score  0.6628
fail_convert_java_cont:6
total_eval_val:0.7941000273665125
-------------------------
数据2048版 1e-5 / e-5
bleu-4 = 0.6309824220215244
rouge_1/f_score  0.5049
rouge_1/r_score  0.4805
rouge_1/p_score  0.5510
rouge_2/f_score  0.3682
rouge_2/r_score  0.3487
rouge_2/p_score  0.4122
rouge_l/f_score  0.5368
rouge_l/r_score  0.5821
rouge_l/p_score  0.6621
fail_convert_java_cont:5
total_eval_val:0.7578695394199609
-------------------------


-------------------------



-----------------------------------------
可以考虑dev用5的 train用4的 data3
list len not big than 4(3 = 4-1)

------------------------------------------------
还不行就减少数据量 1024版本
---------------------------------------------
还不行 1-》2残差
-----------------------------------------------
还不行 就不残差 单个单个学

--------------------
