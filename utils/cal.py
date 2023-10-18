from rouge import rouge
from bleu import  compute_bleu
import json
from tqdm import tqdm
def load_json_data(data_path):
        #print("loading text-score dataset from: \n   {}".format(data_path))
        with open(data_path, 'r') as f:
            data_list = json.load(f)

        return data_list
def baseline_eval():
    code_reply_data = load_json_data("/home/develop/dzl/PreferCodeLlama/baseLlama/out_predict/result_part_base.json")
    epoch = 0
    total_val = 0.0
    rouge_map = {
      "rouge_1/f_score": 0.0,
      "rouge_1/r_score": 0.0,
      "rouge_1/p_score": 0.0,
      "rouge_2/f_score": 0.0,
      "rouge_2/r_score": 0.0,
      "rouge_2/p_score": 0.0,
      "rouge_l/f_score": 0.0,
      "rouge_l/r_score": 0.0,
      "rouge_l/p_score": 0.0
    }
    for data in tqdm(code_reply_data):
        tokens_test_lables = [data['code_lables']]
        tokens_predict = [data['code_reply']]
        tokens_test_lables_list = []
        tokens_test_lables_list.append(tokens_test_lables)
        result = compute_bleu(tokens_test_lables_list, tokens_predict, 4)
        epoch += 1
        total_val += result[0]
        ROUGE = rouge(tokens_test_lables, tokens_predict)
        for (k, v) in ROUGE.items():
            rouge_map[k] += v
    average_val = total_val/epoch
    for (k,v) in rouge_map.items():
        rouge_map[k] = rouge_map[k]/epoch
    print("bleu-4 = " + str(average_val))
    
    for (k, v) in rouge_map.items():
         print('{} {:7.4f}'.format(k, v))
         
def PreferLlama_eval(json_patch):
    code_reply_data = load_json_data(json_patch)
    epoch = 0
    total_val = 0.0
    rouge_map = {
      "rouge_1/f_score": 0.0,
      "rouge_1/r_score": 0.0,
      "rouge_1/p_score": 0.0,
      "rouge_2/f_score": 0.0,
      "rouge_2/r_score": 0.0,
      "rouge_2/p_score": 0.0,
      "rouge_l/f_score": 0.0,
      "rouge_l/r_score": 0.0,
      "rouge_l/p_score": 0.0
    }
    for data in tqdm(code_reply_data):
        tokens_test_lables = [data['code_lables'].strip('\n')]
        tokens_predict = [data['code_reply']]
        tokens_test_lables_list = []
        tokens_test_lables_list.append(tokens_test_lables)
        result = compute_bleu(tokens_test_lables_list, tokens_predict, 4)
        epoch += 1
        total_val += result[0]
        ROUGE = rouge(tokens_test_lables, tokens_predict)
        for (k, v) in ROUGE.items():
            rouge_map[k] += v
    average_val = total_val/epoch
    for (k,v) in rouge_map.items():
        rouge_map[k] = rouge_map[k]/epoch
    print("bleu-4 = " + str(average_val))
    
    for (k, v) in rouge_map.items():
         print('{} {:7.4f}'.format(k, v))
if __name__ == "__main__":
    
    PreferLlama_eval("/home/develop/dzl/PreferCodeLlama/out_predict/result_part.json") 
    PreferLlama_eval("/home/develop/dzl/PreferCodeLlama/out_predict/result_part_frozzeall.json")  
         
         
    # reference_corpus = [["today is a happy day!"]]
    # translation_corpus = ["today is a happy day!"]
    # result = compute_bleu(reference_corpus, translation_corpus, 4)
    
    # print(result[0])
  
    # tokens_test = ["\ntoday i a happy day!"]
    # tokens_predict = ["today is a happy day!"]
    
    # text_test = [' '.join(tokens) for tokens in tokens_test]
    # text_predict = [' '.join(tokens) for tokens in tokens_predict]
    # ROUGE = rouge(text_test, text_predict)  # a dictionary
    # for (k, v) in ROUGE.items():
    #     print('{} {:7.4f}'.format(k, v))
