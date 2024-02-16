#!/bin/bash

# 启动脚本1，放入后台
bash ./predict_partLong.sh 2 2 &

# 启动脚本2，放入后台
bash ./predict_partLong.sh 3 2 &

# 启动脚本3，放入后台
bash ./predict_partLong.sh 5 3 &

bash ./predict_partLong.sh 6 4 &

# 等待所有后台任务完成
wait