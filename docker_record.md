1.测评4batch (1-p)+p X
2.训练 mean(uesr+input) + MLP 还行
2,1 尝试其对比学习 没啥实际提升
2.若不行 训练 mean(uesr+input) 暂时不做
3.训练 mean(uesr) + MLP 一点点
4.训练mean(uesr+input+部分code) + MLP +contrast Style_code部分有效 blue下降
4.训练 W3(W1mean(uesr) +W2mean(uesr+input+部分code)) + (MLP)
----------------------
以上试试若不很差，试试对比学习
