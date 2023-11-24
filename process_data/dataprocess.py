import json
import csv
from collections import defaultdict
import random
import math
csv.field_size_limit(1 * 1024 * 1024) 
open_Java_dir = './Java_programming/response_log.csv'
open_Cpp_dir = './C++_programming/response_log.csv'
open_Csharp_dir = './Csharp_programming/response_log.csv'
open_Python_dir = './Python_programming/response_log.csv'
open_PHP_dir = './PHP_programming/response_log.csv'

restore_Java_dir = './Java_programming/Java_programming.json'
restore_Cpp_dir = './C++_programming/Cpp_programming.json'
restore_Csharp_dir = './Csharp_programming/Csharp_programming.json'
restore_Python_dir = './Python_programming/Python_programming.json'
restore_PHP_dir = './PHP_programming/PHP_programming.json'

filt_ac_Java_dir = './Java_programming/Java_programming_ac.json'
Java_train_dir = './Java_programming/Java_programming_train.json'
Java_dev_dir = './Java_programming/Java_programming_dev.json'
Java_test_dir = './Java_programming/Java_programming_test.json'
train_part_dir = './Java_part_programming50/Java_programming_train.json'
dev_part_dir = './Java_part_programming50/Java_programming_dev.json'
test_part_dir = './Java_part_programming50/Java_programming_test.json'



filt_ac_Cpp_dir = './C++_programming/C++_programming_ac.json'
Cpp_train_dir = './C++_programming/C++_programming_train.json'
Cpp_dev_dir = './C++_programming/C++_programming_dev.json'
Cpp_test_dir = './C++_programming/C++_programming_test.json'

filt_ac_Csharp_dir = './Csharp_programming/Csharp_programming_ac.json'
Csharp_train_dir = './Csharp_programming/Csharp_programming_train.json'
Csharp_dev_dir = './Csharp_programming/Csharp_programming_dev.json'
Csharp_test_dir = './Csharp_programming/Csharp_programming_test.json'

filt_ac_Python_dir = './Python_programming/Python_programming_ac.json'
Python_train_dir = './Python_programming/Python_programming_train.json'
Python_dev_dir = './Python_programming/Python_programming_dev.json'
Python_test_dir = './Python_programming/Python_programming_test.json'

filt_ac_PHP_dir = './PHP_programming/PHP_programming_ac.json'
PHP_train_dir = './PHP_programming/PHP_programming_train.json'
PHP_dev_dir = './PHP_programming/PHP_programming_dev.json'
PHP_test_dir = './PHP_programming/PHP_programming_test.json'


content_dir = "./problem_content.json"
out_content_dir = './programming_problem_content.json'

#从原始编程数据中获取code_status response集合
def get_response_status(json_dir):
    with open(json_dir,'r',encoding="utf-8") as f:
        data_map =json.load(f)
    response_status_map = dict()
    
    for item in data_map:
        if item["response"] not in response_status_map:
            response_status_map[item["response"]] = 1
        else:
            response_status_map[item["response"]] += 1
    for key, value in response_status_map.items():
        print(key)
        print(value)
    print("------------------------------")
    
#判断json编程数据中各个编程语言的记录条数        
def judge_json(json_dir):
    
    with open(json_dir,'r',encoding="utf-8") as f:
        data_map =json.load(f)
    print(json_dir+" have num:"+str(len(data_map)))
    
    lan_id_map = dict()
    lan_id_map["C#"] = 0
    lan_id_map["Java"] = 0
    for item in data_map:
        if item["language"] == "C#":
            lan_id_map["C#"] += 1
        elif item["language"] == "Java":
            lan_id_map["Java"] += 1
            print(item)
            break
        else :
            str_item = item["language"]
            if str_item in lan_id_map:
                lan_id_map[str_item] += 1
            else:
                 lan_id_map[str_item] = 1
    for key,val in lan_id_map.items():
        print(key + " num:"+str(val))
    print("---------------------------")
    
#从原始编程数据中过滤掉不符合条件的记录 COMPILE_ERROR language
def filt_programming_json(json_dir,out_dir,content_dir, language):
    
    with open(json_dir,'r',encoding="utf-8") as f:
        data_map =json.load(f)
    
    with open(content_dir,'r',encoding="utf-8") as f:
        content_problem_map =json.load(f)
        
    print(json_dir+" have num:"+str(len(data_map)))
    print(content_dir+" have num:"+str(len(content_problem_map)))
    all_ = []
    
    unexist_cont = 0
    flit_cont = 0
    COMPILE_ERROR_cont = 0
    for item in data_map:
        if item["language"] == language :
            if item["response"] != "COMPILE_ERROR":
                flit_cont += 1
                all_.append(item)
                if item["problem_id"] not in content_problem_map:
                    unexist_cont += 1
            else: 
                COMPILE_ERROR_cont += 1
         
    print("there have " + str(unexist_cont)+" unexist problem in content_dir")  
    print("there have " + str(flit_cont)+ language + " record") 
    print(language + " COMPILE_ERROR_cont = " + str(COMPILE_ERROR_cont))
    json.dump(
        all_,
        open(out_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )

#从csv文件中获取前10条json记录
def process_csv_to_json_tmp(open_dir,restore_dir = "./java_programming_tmp.json"):
    csv_file = open(open_dir, 'r')
    csv_reader = csv.DictReader(csv_file)
    all_ = []
    for row in csv_reader:
        
        json_str = json.dumps(row)
        #print(json_str)
        
        json_data = json.loads(json_str) 
        json_data["code"] = json_data["code"].replace("\\\"", "\"").replace("\\n", "\n").replace("\\t", "\t")
        json_data["code"] = json_data["code"].replace("\\r", "\r").replace("\\\n", "\\n").replace("\\\r", "\\r").replace("\\\t", "\\t")
        #print("\\\\")
        #json_data["code"] = json_data["code"].replace("\\\\", "\\").replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r").replace('\\"', '\"')
        #print(json_data["code"])
        all_.append(json_data)
        if len(all_) == 10 :
            break
    json.dump(
        all_,
        open(restore_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )
 
#从csv文件中获得json文件          
def process_csv_to_json(open_dir,restore_dir):
    csv_file = open(open_dir, 'r')
    csv_reader = csv.DictReader(csv_file)
    all_ = []
    for row in csv_reader:
        json_str = json.dumps(row)
        json_data = json.loads(json_str)
        json_data["code"] = json_data["code"].replace("\\\"", "\"").replace("\\n", "\n").replace("\\t", "\t")
        json_data["code"] = json_data["code"].replace("\\r", "\r").replace("\\\n", "\\n").replace("\\\r", "\\r").replace("\\\t", "\\t")
        all_.append(json_data)

    json.dump(
        all_,
        open(restore_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )

#从原始编程数据中获取ac的记录
def filt_accepted_programming_json(json_dir, out_dir, content_dir, language):
    
    with open(json_dir,'r',encoding="utf-8") as f:
        data_map =json.load(f)
    
    with open(content_dir,'r',encoding="utf-8") as f:
        content_problem_map =json.load(f)
        
    print(json_dir+" have num:"+str(len(data_map)))
    print(content_dir+" have num:"+str(len(content_problem_map)))
    all_ = []
    std_ac_map = dict()
    unexist_cont = 0
    flit_cont = 0
    min_std_num = 1e9
    max_std_num = 0
    for item in data_map:
        if item["language"] == language :
            if item["response"] == "ACCEPTED":
                flit_cont += 1
                
                all_.append(item)
                
                if item["user_id"] not in std_ac_map:
                    std_ac_map[item["user_id"]] = 0
                std_ac_map[item["user_id"]] += 1
                min_std_num = min(std_ac_map[item["user_id"]], min_std_num)
                max_std_num = max(std_ac_map[item["user_id"]],max_std_num)
                if item["problem_id"] not in content_problem_map:
                    unexist_cont += 1
            
         
    print("there have " + str(unexist_cont)+" unexist problem in content_dir")  
    print("there have " + str(flit_cont)+ language + " record") 
    print("there min_std_num = " + str(min_std_num))
    print("there max_std_num = " + str(max_std_num))
    print("----------------------------------------")
    json.dump(
        all_,
        open(out_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )
    
#分析ac中的数据中的题目总数量、学生数量以及每个学生答题数,答题超过up_cont的学生个数
def analyze_accepted_programming_json(ac_json_dir, up_cont,single_ac_cont = None):
    '''
    up_cont 答题ac数量超过up_cont
    single_ac_cont 单题ac数量不超过single_ac_cont 为None则不分析
    '''
    with open(ac_json_dir,'r',encoding="utf-8") as f:
        data_map =json.load(f)
    
    print(ac_json_dir+" have num:"+str(len(data_map)))
    
    problem_count = 0
    std_count = 0
    up_std_count = 0 #答题ac数量超过up_cont的学生数量
    std_ac_map = dict() #某学生答对题数
    problem_map = dict()
    std_to_pro_map = defaultdict(set)    # {(stu_id,problem_id): (num)}
    stu_submit_full_map = dict() #提交单题目数量超过single_ac_cont次的学生map
    maxx_std2pro_num = 0
    final_need_student_map = dict() #最终符合条件的学生数量
    final_record_count = 0#最终符合条件的记录数量
    for item in data_map:
        if item["problem_id"] not in problem_map:
            problem_map[item["problem_id"]] = 0
            problem_count += 1
            
        if item["user_id"] not in std_ac_map:
            std_ac_map[item["user_id"]] = 0
            std_count += 1
        if (item["user_id"], item["problem_id"]) not in std_to_pro_map:
            std_ac_map[item["user_id"]] += 1
            std_to_pro_map[(item["user_id"], item["problem_id"])] = 0
            
        std_to_pro_map[(item["user_id"], item["problem_id"])]  += 1
        maxx_std2pro_num = max(maxx_std2pro_num, std_to_pro_map[(item["user_id"], item["problem_id"])])
    if single_ac_cont is not None:
        for item in data_map:
            if  std_to_pro_map[(item["user_id"], item["problem_id"])] > single_ac_cont:
                stu_submit_full_map[item["user_id"]] = True
                
    for user_id, num in std_ac_map.items():    
        if num >= up_cont and user_id not in stu_submit_full_map:
            up_std_count += 1
            final_need_student_map[user_id] = True
    for item in data_map:
        if item["user_id"] in final_need_student_map:
            final_record_count += 1
    
    
        
    print("-----------------------------------------------------------------------------")
    print("total problem count = " + str(problem_count))
    print("total student count = " + str(std_count))
    print("accomplete "+str(up_cont)+ f"nums  and less {single_ac_cont} count | student count = " + str(up_std_count))
    print("finally record number = "+str(final_record_count))
    print("maxx_std2pro_num = "+str(maxx_std2pro_num))
    print("-----------------------------------------------------------------------------")


#生成对应语言的训练集、dev和测试集合
def produce_accepted_programming_json(ac_json_dir, train_out_dir, dev_out_dir, test_out_dir, ratio,up_cont,single_ac_cont=None ,need_std_cont = None):
    '''
    ratio train占比
    up_cont 学生答题超过up_cont
    single_ac_cont 学生单题ac数不超过single_up_cont
    need_std_cont 需要的学生数量 全量的话为None
    '''
    with open(ac_json_dir,'r',encoding="utf-8") as f:
        data_map =json.load(f)
    print(ac_json_dir+" have num:"+str(len(data_map)))
    
 
    up_std_count = 0
    std_ac_map = dict() # {(stu_id): (num)} #某个学生的做题总数
    stdApro_map = defaultdict(set)    # {(stu_id,problem_id): (num)} #某个学生的某题的提交次数
    can_student_id2pnum_map = dict() #满足条件的学生id对应的题目数量
    stu_submit_full_map = dict() #提交单题目数量超过single_ac_cont次的学生map
    for item in data_map:
        if (item["user_id"], item["problem_id"]) not in stdApro_map:
            if item["user_id"] not in std_ac_map:
                std_ac_map[item["user_id"]] = 0
            std_ac_map[item["user_id"]] += 1
            stdApro_map[(item["user_id"], item["problem_id"])] = 0
            
        stdApro_map[(item["user_id"], item["problem_id"])]  += 1
    
    if single_ac_cont is not None:
        for item in data_map:
            if  stdApro_map[(item["user_id"], item["problem_id"])] > single_ac_cont:
                stu_submit_full_map[item["user_id"]] = True
                
    
    std_cont_in_train_map = dict()           # stu_id被选入训练集的问题的个数
    std_cont_in_dev_map = dict()           # stu_id被选入dev的问题的个数              
    train_stdApro_map = defaultdict(set)    # {(stu_id,problem_id): (1)} 已选入train中的学生（stu_id）的题号problem_id
    dev_stdApro_map = defaultdict(set)
    train = []
    dev = []
    test = []
    #过滤得到答题数大于等于up_cont的学生id
    #二次过滤学生数量
    now_std_num = 0
    for user_id, num in std_ac_map.items(): 
        if single_ac_cont is not None:
            if user_id in stu_submit_full_map:
                continue
            
        if num >= up_cont:
            up_std_count += 1
            can_student_id2pnum_map[user_id] = num
            std_cont_in_train_map[user_id] = 0
            now_std_num += 1
            std_cont_in_dev_map[user_id] = 0
        if need_std_cont is not None:
            if now_std_num == need_std_cont:
                break            
    print("now_std_num (statis up_cont and single number) = " + str(now_std_num))
    
    record_number = 0
    for item in data_map:
        user_id = item["user_id"]
        problem_id = item["problem_id"]
        if user_id not in std_cont_in_train_map: #过滤小于upcont的学生id 过滤单题ac提交数超过 single_ac_cont
            continue
        #if user_id in stu_submit_full_map: #
        #    continue
        newitem ={"user_id":item['user_id'],"problem_id":item['problem_id'],"code":item['code']}
        item = newitem
        record_number += 1
        if (user_id, problem_id) in train_stdApro_map:
            train.append(item) 
            continue
        
        if (user_id, problem_id) in dev_stdApro_map:
            dev.append(item) 
            continue
        
        if std_cont_in_train_map[user_id]  <  can_student_id2pnum_map[user_id] * ratio:
            train_stdApro_map[(user_id, problem_id)] = 1
            std_cont_in_train_map[user_id] += 1
            train.append(item) 
            
                 
        elif std_cont_in_dev_map[user_id] < can_student_id2pnum_map[user_id] * ((1-ratio)/2.0):
            dev_stdApro_map[(user_id, problem_id)] = 1
            std_cont_in_dev_map[user_id] += 1
            dev.append(item)
            
        else:
            test.append(item)
    
    print(f"totoal record = {record_number}")        
    print("total train count = " + str(len(train)))
    print("total dev count = " + str(len(dev)))
    print("total test count = " + str(len(test)))
    

    json.dump(
        train,
        open(train_out_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )
    json.dump(
        dev,
        open(dev_out_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )
    json.dump(
        test,
        open(test_out_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )


def split_accepted_programming_json(ac_json_dir, train_out_dir, dev_out_dir, test_out_dir, ratio,up_cont,single_ac_cont=None ,need_std_cont = None):
    
    with open(ac_json_dir,'r',encoding='utf-8') as ac_file:
        ac_data=json.load(ac_file)
    
    forbidden_std= set()   # 不符合条件的学生集合
    std_ac_map = dict() # {(stu_id): (num)} #某个学生的做题总数
    stdApro_map = defaultdict(set)    # {(stu_id,problem_id): (num)} #某个学生的某题的提交次数
    stu_submit_full_map = dict() #提交单题目数量超过single_ac_cont次的学生map
    for item in ac_data:
        if (item["user_id"], item["problem_id"]) not in stdApro_map:
            if item["user_id"] not in std_ac_map:
                std_ac_map[item["user_id"]] = 0
            std_ac_map[item["user_id"]] += 1
            stdApro_map[(item["user_id"], item["problem_id"])] = 0
            
        stdApro_map[(item["user_id"], item["problem_id"])]  += 1
    
    if single_ac_cont is not None:
        for item in ac_data:
            if  stdApro_map[(item["user_id"], item["problem_id"])] > single_ac_cont:
                stu_submit_full_map[item["user_id"]] = True
    
    # 根据 1.某人做题总数 2.某人某题ac上限 剔除部分学生
    for user_id, num in std_ac_map.items(): 
        if single_ac_cont is not None:
            if user_id in stu_submit_full_map:
                forbidden_std.add(user_id)
                continue
        if num < up_cont:
            forbidden_std.add(user_id)
            continue
    
    # 去除不合格学生
    choose_std_set=set()
    for user_id, num in std_ac_map.items():       
        if user_id in forbidden_std:
            continue
        choose_std_set.add(user_id)
    
    
    # (A,B,C,D,E,F)-> (A,B,C,D) (E) (F)
    problem_pool=set()
   
    for item in ac_data:
        if item['user_id'] in choose_std_set:
            problem_pool.add(item['problem_id'])
       
    problem_list = list(problem_pool)
    random.shuffle(problem_list)
    problem_list_len = len(problem_list)
    dev_problem_size = math.floor(problem_list_len*(1-ratio)/2)
    test_problem_size = math.floor(problem_list_len*(1-ratio)/2)
    dev_problem = set(problem_list[:dev_problem_size])
    test_problem = set(problem_list[dev_problem_size:dev_problem_size+test_problem_size])
               
    train = []
    dev = []
    test = []
    
    in_train_std = set()
    in_dev_std=set()
    in_test_std=set()    
    for item in ac_data:
        if item['user_id'] not in choose_std_set:
            continue 
        if item['problem_id'] in dev_problem:
            in_dev_std.add(item['user_id'])
        elif item['problem_id'] in test_problem:
            in_test_std.add(item['user_id'])
        else:
            in_train_std.add(item['user_id'])
            
    final_choose_stds=set()
    
    for std in choose_std_set:
        if std in in_test_std and std in in_train_std and std in in_dev_std:
            final_choose_stds.add(std)
            if need_std_cont is not None: 
                if len(final_choose_stds)==need_std_cont:
                    break 
        
    in_dev_std.clear()
    in_test_std.clear()
    in_train_std.clear()
    
    for item in ac_data:
        if item['user_id'] not in final_choose_stds:
            continue 
        if item['problem_id'] in dev_problem:
            dev.append(item)
            in_dev_std.add(item['user_id'])
        elif item['problem_id'] in test_problem:
            test.append(item)
            in_test_std.add(item['user_id'])
        else:
            in_train_std.add(item['user_id'])
            train.append(item)
            
    print('total user choose'+str(len(choose_std_set)))
    print('total user really train'+str(len(in_train_std) ))       
    print('total user really dev'+str(len(in_dev_std)))    
    print('total user really test'+str(len(in_test_std)))                   
    print("total train count = " + str(len(train)))
    print("total dev count = " + str(len(dev)))
    print("total test count = " + str(len(test)))
    print("totoal data count = " + str(len(test) + len(dev) + len(train)))

    json.dump(
        train,
        open(train_out_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )
    json.dump(
        dev,
        open(dev_out_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )
    json.dump(
        test,
        open(test_out_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )
    
#手动设置比例和答题超过指定数量的学生数, 新加参数指定学生数
def process_to_json(content_dir, lanauage, open_csv_dir, restore_json_dir, filt_ac_dir, train_dir, dev_dir, test_dir):
    process_csv_to_json(open_csv_dir, restore_json_dir)
    filt_accepted_programming_json(restore_json_dir, filt_ac_dir, content_dir, lanauage)
    analyze_accepted_programming_json(filt_ac_dir,20, 1)
    #produce_accepted_programming_json(filt_ac_dir,train_dir,dev_dir,test_dir, 0.8, 20,1, None)
    split_accepted_programming_json(filt_ac_dir,train_dir,dev_dir,test_dir, 0.8, 20,1, None) 
    
            
#导出指定problem_id的记录           
def  get_problem_json(content_dir, id_list, out_content_dir):
    with open(content_dir,'r',encoding="utf-8") as f:
        content_problem_map =json.load(f)
    all_ = []
    for id in id_list:
        item = {"id":id,'content':content_problem_map[id]}
        print(content_problem_map[id])
        all_.append(item)
    json.dump(
        all_,
        open(out_content_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )        


def remove_control_characters(input_str):
    # 创建一个包含所有控制字符的字符串
    control_chars = ''.join([ch for ch in string.printable if ch not in string.printable[:-5]])
    
    # 使用str.translate()方法将控制字符替换为空字符
    translation_table = str.maketrans('', '', control_chars)
    cleaned_str = input_str.translate(translation_table)
    
    return cleaned_str

def remove_control_chars(text):
    # 使用str.translate()删除控制字符
    control_chars = ''.join(map(chr, range(0, 32)))  # 创建控制字符的字符串
    control_chars_translation = str.maketrans('', '', control_chars)
    cleaned_text = text.translate(control_chars_translation)
    return cleaned_text

 
def form_ac_json_get_use_content(filt_ac_Java_dir, filt_ac_Csharp_dir, filt_ac_Cpp_dir, filt_ac_Python_dir, content_dir):
    id_map = dict()
    with open(content_dir,'r',encoding="utf-8-sig") as f:
        content_problem_map =json.load(f)
    print("origin problem content num = " + str(len(content_problem_map)))
    
    
    with open(filt_ac_Cpp_dir,'r',encoding="utf-8") as f:
        filt_ac_Cpp_dir =json.load(f)
    for item in filt_ac_Cpp_dir:
        p_id = item["problem_id"]
        id_map[p_id] = 1 
    
    with open(filt_ac_Java_dir,'r',encoding="utf-8") as f:
        filt_ac_Java_dir =json.load(f)
    for item in filt_ac_Java_dir:
        p_id = item["problem_id"]
        id_map[p_id] = 1 
    
    with open(filt_ac_Csharp_dir,'r',encoding="utf-8") as f:
        filt_ac_Csharp_dir =json.load(f)
    for item in filt_ac_Csharp_dir:
        p_id = item["problem_id"]
        id_map[p_id] = 1  
        
    with open(filt_ac_Python_dir,'r',encoding="utf-8") as f:
        filt_ac_Python_dir =json.load(f)
    for item in filt_ac_Python_dir:
        p_id = item["problem_id"]
        id_map[p_id] = 1 
          

    print("programming num = " + str(len(id_map)))         
    all_ = []
    for id,content in content_problem_map.items():
        if id in id_map:
            all_.append({"id":id,'content':content})


    print("content programming num = " + str(len(all_)))    
    json.dump(
        all_,
        open("./programming_problem_content_tmp.json", 'w'),
        indent=4,
        ensure_ascii=False,
    )
        
    

def  test_out_len(content_dir, id_list, out_content_dir):
    with open(content_dir,'r',encoding="utf-8") as f:
        content_problem_map =json.load(f)
    all_ = []
    for id in id_list:
        item = {"id":id,'content':content_problem_map[id]}
        print(content_problem_map[id])
        all_.append(item)
    print(len(all_))
    json.dump(
        all_,
        open(out_content_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )    


def test_format_process_csv_to_json(open_dir,restore_dir = "./java_programming_format.json"):
    csv_file = open(open_dir, 'r')
    csv_reader = csv.DictReader(csv_file)
    all_ = []
    for row in csv_reader:
        json_str = json.dumps(row)
        json_data = json.loads(json_str) 
        if json_data["language"] != "Java" : continue
        #if json_data["submission_id"] != "1274520912634765312": continue
        json_data["code"] = json_data["code"].replace("\\\"", "\"").replace("\\n", "\n").replace("\\t", "\t")
        json_data["code"] = json_data["code"].replace("\\r", "\r").replace("\\\n", "\\n").replace("\\\r", "\\r").replace("\\\t", "\\t")
        all_.append(json_data)
        if len(all_) == 120  :
            break
    json.dump(
        all_,
        open(restore_dir, 'w'),
        indent=4,
        ensure_ascii=False
    )   
     
if __name__ == "__main__":
    # 示例输入字符串
    #random.seed(42)
    #ls = list()
    #ls.append('1074931048055713792')
    #get_problem_json(content_dir,ls,'./delete.json')

    #process_to_json(content_dir,"C++",open_Cpp_dir,restore_Cpp_dir,filt_ac_Cpp_dir,Cpp_train_dir,Cpp_dev_dir,Cpp_test_dir)
    #process_to_json(content_dir,"Java",open_Java_dir,restore_Java_dir,filt_ac_Java_dir,Java_train_dir,Java_dev_dir,Java_test_dir)
    #process_to_json(content_dir,"C#",open_Csharp_dir,restore_Csharp_dir,filt_ac_Csharp_dir,Csharp_train_dir,Csharp_dev_dir,Csharp_test_dir)
    #process_to_json(content_dir,"Python",open_Python_dir,restore_Python_dir,filt_ac_Python_dir,Python_train_dir,Python_dev_dir,Python_test_dir)
   
    #form_ac_json_get_use_content(filt_ac_Java_dir, filt_ac_Csharp_dir, filt_ac_Cpp_dir, filt_ac_Python_dir, content_dir)
    #process_to_json(content_dir,"Java",open_Java_dir,restore_Java_dir,filt_ac_Java_dir,Java_train_dir,Java_dev_dir,Java_test_dir)
    #produce_accepted_programming_json(filt_ac_Java_dir,train_part_dir,dev_part_dir,test_part_dir, 0.8, 20, 1, 10) #100个学生
     
     
    #analyze_accepted_programming_json(filt_ac_Java_dir,20, 1)
    #split_accepted_programming_json(filt_ac_Java_dir,Java_train_dir,Java_dev_dir,Java_test_dir, 0.8, 20, 1, None) #全量
    split_accepted_programming_json(filt_ac_Java_dir,train_part_dir,dev_part_dir,test_part_dir, 0.8, 20, 1, 50) # 50学生
    #split_accepted_programming_json(filt_ac_Cpp_dir,Cpp_train_dir,Cpp_dev_dir,Cpp_test_dir, 0.8, 20,1, None) 
    #test_format_process_csv_to_json(open_Java_dir)
    #input()
    #analyze_accepted_programming_json(filt_ac_Java_dir,20, 1)