import json
import subprocess
import time
import xml.etree.ElementTree as ET
import numpy as np
import numpy as np
import scipy.stats
import io
import os
from tqdm import tqdm
prefer_predict_json_path = "./data/Long50_MPCode_25.json"
base_predict_json_patch = "./data/Long50_MSAAdapterM55_noScaler.json"

real_out_dir = "./prefer_predict_result/real_result_list_Long50.json"
#base_out_dir = "./base_predict_result/data_MSAAdapter_test.json"
prefer_out_dir = "./prefer_predict_result/Long50_MPCode_25.json"

# prefer_predict_json_path = "./data/softprompt_Short1121.json"
# base_predict_json_patch = "./data/short1121_base_result.json"

# real_out_dir = "./base_predict_result/short1121_real_result_list.json"
# base_out_dir = "./base_predict_result/short1121_base_result.json"
# prefer_out_dir = "./prefer_predict_result/softprompt_Short1121.json"
def inint_style_map():
    initial_value = 0
    keys = ['RightCurly','SeparatorWrap','NoLineWrapCheck', 'AvoidStarImportCheck', 'OneTopLevelClassCheck',
            'EmptyLineSeparatorCheck', 'WhitespaceAroundCheck', 'GenericWhitespaceCheck',
            'OperatorWrapCheck','LineLengthCheck','LeftCurlyCheck', 'EmptyBlockCheck',
            'NeedBracesCheck', 'MultipleVariableDeclarationsCheck',
            'OneStatementPerLineCheck','UpperEllCheck', 'ModifierOrderCheck', 
            'FallThroughCheck','MissingSwitchDefaultCheck', 
            'TypeNameCheck', 'MethodNameCheck','MemberNameCheck', 'ParameterNameCheck', 'LocalVariableNameCheck']
    style_map = {key: initial_value for key in keys}
    return style_map

#first针对code_lables生成完整的numpy文件
def load_json_data(data_path):
        with open(data_path, 'r') as f:
            data_list = json.load(f)

        return data_list

# 要执行的Shell命令
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
        #print(str(stderr_output))
        #input()
        if len(str(stderr_output)) != 2:
            #print("hhh")
            return False
        return True
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return False
    #print("Script terminated.")

def generte_style_result(result_xml, style_map):
    # 加载 XML 文件
    if os.path.exists(result_xml) == False or os.path.getsize(result_xml) <= 0:
        result_map = {key: 0 for key,value in style_map.items()}
        style_list = []
        for key,value in result_map.items():
            style_list.append(value)
        return style_list
    
    tree = ET.parse(result_xml)
    root = tree.getroot()
    file_xml = root.find("file")
    # 遍历 XML 树
    result_map = {key: 0 for key,value in style_map.items()}
    for item in file_xml:
        text = item.attrib["source"].split('.')[-1]
        if text in result_map:
            result_map[text] += 1
        elif 'RightCurly' in text :
            result_map['RightCurly'] += 1
        elif 'SeparatorWrap' in text :
            result_map['SeparatorWrap'] += 1
    
    style_list = []
    for key,value in result_map.items():
        style_list.append(value)
    return style_list


def Create_file(java_code, file_name):
    # 写入文件
    with open(file_name, "w") as file:
        file.write(java_code)
    #print(f"Java文件已创建：{file_name}")
    
def get_Style_result(code, idx, style_map):
    #1.生成java文件
    file_name = f"java_file/main{idx}.java"
  
    Create_file(code.replace('\\\\\"', '\\"'), file_name)
    
    #2.用命令行执行javawenjian来生成对应xml文件
    your_shell_command = f"java -Djava.io.tmpdir=./ -jar  ./checkstyle-10.12.4-all.jar -c ./google_checks.xml ./java_file/main{idx}.java -f xml -o ./xml_file/result{idx}.xml "
    result_xml = f"./xml_file/result{idx}.xml"
    
    flag = run_shell_command(your_shell_command)
    
    #3.解析xml文件得到
    if flag == True:
        style_list = generte_style_result(result_xml, style_map)
    else:
        style_list = [0 for i in range(len(style_map))]
    
    
    return style_list, flag

def get_predict_java_list(predict_json_path,is_true, out_dir):
    predict_json_data_list =  load_json_data(predict_json_path) 
    idx = 1
    style_map = inint_style_map()
    result_json_path = []
    fail_json_path = []
    fail_convert_java_cont = 0
    num = 0
    for item in tqdm(predict_json_data_list):
        if is_true:
            code = item["code_lables"]
            #print(code)
        else:
            code = item["code_reply"]
           
        item_result_list,flag = get_Style_result(code, idx, style_map)
        if flag == False:
            fail_convert_java_cont += 1
            fail_json_path.append({'problem_id':item["problem_id"],'user_id':item["user_id"],"code_reply":item["code_reply"]})
        result_json_path.append({'problem_id':item["problem_id"],'user_id':item["user_id"],"flag":flag,"result_list":item_result_list})
        #idx += 1 复用文件
        num += 1
        #print(num)
        if num%1000 == 0:
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
    json.dump(
                fail_json_path,
                open("./fail_java.json", 'w'),
                indent=4,
                ensure_ascii=False
            )
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!fail_convert_java_cont:"+str(fail_convert_java_cont))

def JS_divergence(p,q):
    M=(p+q)/2
    return 0.5*scipy.stats.entropy(p,M)+0.5*scipy.stats.entropy(q, M)

def get_filt_result_distribution(result_list,style_map,new_map):
    idx = 0
    style_list = []
    for key,value in style_map.items():
        if key in new_map:
            style_list.append(result_list[idx])
        idx += 1
    
    return style_list

def eval_style_sim(real_distribution_path, predict_distribution_path):
    real_distribution_map = load_json_data(real_distribution_path)
    predict_distribution_map = load_json_data(predict_distribution_path)
    real_distribution_list = []
    predict_distribution_list = []
    correct_list = []
    real_correct_list = []
    
    #map1 = inint_style_map()
    #map2 = inint_new_style_map()  
    for item in real_distribution_map:
        input_item = item["result_list"]
        #input_item = get_filt_result_distribution(item["result_list"],map1,map2)
        real_distribution_list.append(input_item)
        real_correct_list.append(item["flag"])
    for item in predict_distribution_map:
        input_item = item["result_list"]
        #input_item = get_filt_result_distribution(item["result_list"],map1,map2)
        predict_distribution_list.append(input_item)
        correct_list.append(item["flag"])
    if len(real_distribution_list) != len(predict_distribution_list):
        print("ERROR SIZE!!")
        return 
    record_len = len(real_distribution_list)
    total_eval_val = 0.0
    eps = 1e-10
    record_num = 0
    fail_num = 0
    for i in range(record_len):
        real_list = real_distribution_list[i]
        predict_list = predict_distribution_list[i]
        
        if correct_list[i] == False or real_correct_list[i] == False: 
            fail_num += 1    
            #print("00.00")
            continue
        record_num += 1
        
        real_numpy = np.array(real_list)
        real_numpy = real_numpy + eps
        array_sum = np.sum(real_numpy)
        real_probability_vector = real_numpy / array_sum
        
        predict_numpy = np.array(predict_list)    
        predict_numpy = predict_numpy + eps
        array_sum = np.sum(predict_numpy)
        predict_probability_vector = predict_numpy / array_sum
        #print(real_probability_vector)
        #print(predict_probability_vector)
        sim_val = 1-JS_divergence(real_probability_vector, predict_probability_vector)
        #print(sim_val)
        total_eval_val += sim_val
    r_eval_val = total_eval_val / record_num
    new_eval_val = total_eval_val / (record_num+fail_num)
    print("fail_num:" + str(fail_num))
    print("record_num:" + str(record_num))
    print("r_eval_val:" + str(r_eval_val))
    print("new_eval_val:" + str(new_eval_val))
def contorl_error_not_stop():
    idx = 1
    command = f"java -jar ./checkstyle-10.12.4-all.jar -c ./google_checks.xml ./java_file/main{idx}.java -f xml -o ./xml_file/result{idx}.xml"  # 以 "ls -l" 为例
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 获取标准输出和标准错误输出
    stdout_output = result.stdout
    stderr_output = result.stderr

    # 打印输出
    print("Standard Output:")
    #print(stdout_output)
    print("stdout_output:" + str(len(stdout_output)))
    print("\nStandard Error Output:")
    #print(stderr_output)
    print("stderr_output:" + str(len(stderr_output)))
    
if __name__ == "__main__":
    #get_predict_java_list(base_predict_json_patch,True,real_out_dir)
    
    #get_predict_java_list(base_predict_json_patch,False,base_out_dir)
    #eval_style_sim(real_out_dir, base_out_dir)
    
    get_predict_java_list(prefer_predict_json_path,False,prefer_out_dir)
    eval_style_sim(real_out_dir, prefer_out_dir)
    
    # result = [ 0 for i in range(0,25)]
    # map1 = inint_style_map()
    # map2 = inint_new_style_map()
    # s = get_filt_result_distribution(result, map1, map2)
    # print(len(s))
    #str_base ="public class Student {\n    private int id;\n    private String name;\n    private String gender;\n    private String major;\n    private int score;\n\n    public Student(int id, String name, String gender, String major, int score) {\n        this.id = id;\n        this.name = name;\n        this.gender = gender;\n        this.major = major;\n        this.score = score;\n    }\n\n    public int getId() {\n        return id;\n    }\n\n    public void setId(int id) {\n        this.id = id;\n    }\n\n    public String getName() {\n        return name;\n    }\n\n    public void setName(String name) {\n        this.name = name;\n    }\n\n    public String getGender() {\n        return gender;\n    }\n\n    public void setGender(String gender) {\n        this.gender = gender;\n    }\n\n    public String getMajor() {\n        return major;\n    }\n\n    public void setMajor(String major) {\n        this.major = major;\n    }\n\n    public int getScore() {\n        return score;\n    }\n\n    public void setScore(int score) {\n        this.score = score;\n    }\n\n    public abstract String getGrade();\n}\n\npublic class Undergraduate extends Student {\n    public Undergraduate(int id, String name, String gender, String major, int score) {\n        super(id, name, gender, major, score);\n    }\n\n    @Override\n    public String getGrade() {\n        if (score >= 80 && score <= 100) {\n            return \"A\";\n        } else if (score >= 70 && score <= 80) {\n            return \"B\";\n        } else if (score >= 60 && score <= 70) {\n            return \"C\";\n        } else if (score >= 50 && score <= 60) {\n            return \"D\";\n        } else {\n            return \"E\";\n        }\n    }\n}\n\npublic class Graduate extends Student {\n    private String supervisor;\n\n    public Graduate(int id, String name, String gender, String major, String supervisor, int score) {\n        super(id, name, gender, major, score);\n        this.supervisor = supervisor;\n    }\n\n    @Override\n    public String getGrade() {\n        if (score >= 90 && score <= 100) {\n            return \"A\";\n        } else if (score >= 80 && score <= 90) {\n            return \"B\";\n        } else if (score >= 70 && score <= 80) {\n            return \"C\";\n        } else if (score >= 60 && score <= 70) {\n            return \"D\";\n        } else {\n            return \"E\";\n        }\n    }\n}\n\npublic class Main {\n    public static void main(String[] args) {\n        Undergraduate undergraduate = new Undergraduate(2, \"chen\", \"female\", \"cs\", 90);\n        Graduate graduate = new Graduate(3, \"li\", \"male\", \"sc\", \"wang\", 80);\n\n        System.out.println(undergraduate.getGrade());\n        System.out.println(graduate.getGrade());\n    }\n}"
    #Create_file(str_base,"./base_exmaple.java")
   