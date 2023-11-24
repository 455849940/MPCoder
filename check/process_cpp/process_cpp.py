import time 
import numpy as  np
import json
import subprocess
import scipy.stats
base_predict_json_patch = "./data/result_part_base100.json"

real_out_dir = "./prefer_predict_result/real_result_list.json"
base_out_dir = "./base_predict_result/base_result_list.json"

def Create_file(cpp_code, file_name):
    with open(file_name, "w") as file:
        file.write(cpp_code)

_ERROR_CATEGORIES = [
    'build/class',
    'build/c++11',
    'build/c++14',
    'build/c++tr1',
    'build/deprecated',
    'build/endif_comment',
    'build/explicit_make_pair',
    'build/forward_decl',
    'build/header_guard',
    'build/include',
    'build/include_subdir',
    'build/include_alpha',
    'build/include_order',
    'build/include_what_you_use',
    'build/namespaces_headers',
    'build/namespaces_literals',
    'build/namespaces',
    'build/printf_format',
    'build/storage_class',
    'legal/copyright',
    'readability/alt_tokens',
    'readability/braces',
    'readability/casting',
    'readability/check',
    'readability/constructors',
    'readability/fn_size',
    'readability/inheritance',
    'readability/multiline_comment',
    'readability/multiline_string',
    'readability/namespace',
    'readability/nolint',
    'readability/nul',
    'readability/strings',
    'readability/todo',
    'readability/utf8',
    'runtime/arrays',
    'runtime/casting',
    'runtime/explicit',
    'runtime/int',
    'runtime/init',
    'runtime/invalid_increment',
    'runtime/member_string_references',
    'runtime/memset',
    'runtime/indentation_namespace',
    'runtime/operator',
    'runtime/printf',
    'runtime/printf_format',
    'runtime/references',
    'runtime/string',
    'runtime/threadsafe_fn',
    'runtime/vlog',
    'whitespace/blank_line',
    'whitespace/braces',
    'whitespace/comma',
    'whitespace/comments',
    'whitespace/empty_conditional_body',
    'whitespace/empty_if_body',
    'whitespace/empty_loop_body',
    'whitespace/end_of_line',
    'whitespace/ending_newline',
    'whitespace/forcolon',
    'whitespace/indent',
    'whitespace/line_length',
    'whitespace/newline',
    'whitespace/operators',
    'whitespace/parens',
    'whitespace/semicolon',
    'whitespace/tab',
    'whitespace/todo',
    ]

_head_errors=[
    'whitespace/tab',
    'whitespace/operators',
    'whitespace/braces',
    'whitespace/comma',
    'whitespace/indent',
    'whitespace/comments',
    'whitespace/end_of_line',
    'whitespace/parens',
    'whitespace/semicolon',
    'build/namespaces',
    'whitespace/ending_newline',
    'whitespace/blank_line',
    'readability/braces',
    'whitespace/line_length',
    'whitespace/newline',
    'runtime/explicit',
    'runtime/references',
    'runtime/int',
    'runtime/arrays',
    'runtime/string',
    ]


def statistic(path):
    stytle_dict = {}
    with open(path,'r',encoding='utf-8') as file:
        data = json.load(file)
        for idx,item in enumerate(data):
            code = item['code']
            file_name = f"cpp_file/main{idx}.cpp"
            Create_file(code.replace('\\\\\"', '\\"'), file_name)
            shell_cmd=f"cpplint {file_name} > text.txt 2>&1"
            result = subprocess.run(shell_cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with open('text.txt','r') as file:
                for line in file: 
                    for item in _ERROR_CATEGORIES:
                        if line.find(item) !=-1:
                            if item in stytle_dict:
                                stytle_dict[item]+=1
                            else:
                                stytle_dict[item]=1    
    return stytle_dict

def pre_collect():
    path_train  = 'cpp_file/raw_data/C++_programming_train.json'
    path_dev  = 'cpp_file/raw_data/C++_programming_dev.json'
    path_test  = 'cpp_file/raw_data/C++_programming_test.json'
    
    dict_train = statistic(path_train)
    dict_dev = statistic(path_dev)
    dict_test = statistic(path_test)
    
    merged_dict = {key: value for d in (dict_train, dict_dev, dict_test) for key, value in d.items()}
    sorted_dict = dict(sorted(merged_dict.items(), key=lambda item: item[1],reverse=True))
    
    out_path= 'max_type.txt'
    with open(out_path,'w',encoding='utf-8') as file:
        for key,val in sorted_dict.items():
            out=key+"-----"+str(val)+'\n'
            file.writelines(out)
            
def run_shell_command(your_shell_command):
    try:
        # 启动Shell命令
        process = subprocess.Popen(your_shell_command, shell=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
        process.wait()
        #result = subprocess.run(your_shell_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #result.wait()
        # 获取标准输出和标准错误输出
        #stdout_output = process.stdout
        #print(process.stderr)
        stderr_output = process.stderr.readlines()
        
        if len(str(stderr_output)) != 2:
            #print("hhh")
            return False
        return True
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False




def inint_style_map():
    initial_value = 0
    style_map = {key: initial_value for key in _head_errors}
    return style_map


def statistic_style(middle_file_name,style_map):
    with open(middle_file_name,'r',encoding='utf-8') as file:
        for line in file:
            for item in _head_errors:
                if line.find(item) !=-1:
                    style_map[item]+=1
    
    style_list=[]                
    for key,val in style_map.items():
        style_list.append(val)
    
    return style_list

def get_Style_result(code, idx, style_map):
    # 1.generate cpp file
    file_name = f"cpp_file/main{idx}.cpp"
    middle_file_name = 'text.txt'
    Create_file(code.replace('\\\\\"', '\\"'), file_name)
    
    # 2. run shell and get .txt file
    shell_cmd=f"cpplint {file_name} > {middle_file_name} 2>&1"
    run_shell_command(shell_cmd)
    
    # 3. statistic style
    style_list = statistic_style(middle_file_name,style_map)    
    return style_list, True   


def load_json_data(data_path):
    with open(data_path, 'r') as f:
        data_list = json.load(f)

    return data_list

def get_predict_cpp_list(predict_json_path,is_true, out_dir):
    predict_json_data_list = load_json_data(predict_json_path) 
    idx = 1
    result_json_path = []
    fail_convert_java_cont = 0
    num = 0
    for item in predict_json_data_list:
        style_map = inint_style_map()
        if is_true:
            code = item["code_lables"]
        else:
            code = item["code_reply"]
        item_result_list,flag = get_Style_result(code, idx, style_map)
        if flag == False:
            fail_convert_java_cont += 1
        result_json_path.append({'problem_id':item["problem_id"],'user_id':item["user_id"],"flag":flag,"result_list":item_result_list})
        #idx += 1 复用文件
        num += 1
        if num%100 == 0:
            json.dump(
                result_json_path,
                open(out_dir, 'w'),
                indent=4,
                ensure_ascii=False
            )
    json.dump(
                result_json_path,
                open(out_dir, 'w'),
                indent=4,
                ensure_ascii=False
            )
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!fail_convert_cpp_cont:"+str(fail_convert_java_cont))

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p,M)+0.5*scipy.stats.entropy(q, M) 

def eval_style_sim(real_distribution_path, predict_distribution_path):
    real_distribution_map = load_json_data(real_distribution_path)
    predict_distribution_map = load_json_data(predict_distribution_path)
    real_distribution_list = []
    predict_distribution_list = []
    correct_list = []
    for item in real_distribution_map:
        real_distribution_list.append(item["result_list"])
    for item in predict_distribution_map:
        predict_distribution_list.append(item["result_list"])
        correct_list.append(item["flag"])
    record_len = len(real_distribution_list)
    total_eval_val = 0.0
    eps = 1e-10
    for i in range(record_len):
        real_list = real_distribution_list[i]
        predict_list = predict_distribution_list[i]

        if correct_list[i] == False: continue
            
        real_numpy = np.array(real_list)
        real_numpy = real_numpy + eps
        array_sum = np.sum(real_numpy)
        real_probability_vector = real_numpy / array_sum
        
        predict_numpy = np.array(predict_list)    
        predict_numpy = predict_numpy + eps
        array_sum = np.sum(predict_numpy)
        predict_probability_vector = predict_numpy / array_sum
        sim_val = 1-JS_divergence(real_probability_vector, predict_probability_vector)
        print(sim_val)
        print("________________________________________________________________________________")
        total_eval_val += sim_val
    total_eval_val = total_eval_val / record_len
    print("total_eval_val:" + str(total_eval_val))          
                    
if __name__ == "__main__": 
    get_predict_cpp_list(base_predict_json_patch,True,real_out_dir)
    get_predict_cpp_list(base_predict_json_patch,False,base_out_dir)
    
    eval_style_sim(real_out_dir, base_out_dir)

            