#!/bin/bash

# 检查参数是否提供
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_iterations>"
    exit 1
fi

# 从命令行参数获取循环次数
iterations="$1"

# 循环执行 delete_file.sh
for ((i=1; i<=iterations; i++)); do
    bash human_eval.sh "$i"
    echo "------------------------"
done
