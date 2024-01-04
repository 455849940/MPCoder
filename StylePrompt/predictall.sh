#!/bin/bash

# 启动脚本1，放入后台
bash predict1.sh > ./script/1.log &

# 启动脚本2，放入后台
bash  predict2.sh > ./script/2.log &

# 启动脚本3，放入后台
bash  predict3.sh > ./script/3.log  &

bash  predict4.sh > ./script/4.log &

# 等待所有后台任务完成
wait