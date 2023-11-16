from typing import Dict, Optional, Sequence, List
import json

def load_json_data(data_path):
    #print("loading text-score dataset from: \n   {}".format(data_path))
    data_list = []
    with open(data_path, 'r') as file:
        for line in file:
            # 逐行解析JSON对象
            data = json.loads(line)
            data_list.append(data)
    return data_list

def process_humaneval_data(data_json_path, human_eval_out_path):
    data_list = load_json_data(data_json_path)
    generation_json = []
    for data in data_list:
        # print(data)
        function_name = data['prompt'].split("public")[-1]
        
        function_name = "public"+function_name
        # print(function_name)
        function_body = data['generation'].split(function_name)
        is_have_solution = False
        if "Solution" in data['generation']:
            is_have_solution = True
        item = data
        if(len(function_body) != 2):
            
            # print(function_body)
            # print(function_name)
            # print(data)
            # input()
            # # print("--------------")
            # print(1)
            item['generation'] = function_body[0]
        elif is_have_solution == False:
            item['generation'] = function_body[1] + '\n}'
        else:
            item['generation'] = function_body[1]
        # print(is_have_solution)
        # print("-----------------")  
        # print(item['generation'])        
        # print("----------------------")
        # input()
        
        generation_json.append(item)
    
    
    with open(human_eval_out_path, 'w') as file:
        for item in generation_json:
            json.dump(item, file)
            file.write('\n')   

if __name__ == "__main__":
    #process_humaneval_data("../out_predict/humeval_result_aug_model_3.jsonl","../humaneval_data/aug_model_humaneval_data.jsonl")
    process_humaneval_data("../humaneval_data/humeval_result_style_model_data.jsonl","../humaneval_data/Style_model_humaneval_data.jsonl")