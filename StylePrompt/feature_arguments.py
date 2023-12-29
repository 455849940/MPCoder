from typing import List, Optional

from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class train_config(TrainingArguments):
    
    continue_train: bool = field(
        default= False,
        metadata={"help": "continue train."}
    )
    
    forwardChoose: int = field(
        default= 0, 
        metadata={"help": "forwardChoose [0,1]"}
    )
    
    forwardChoose2: int = field(
        default= 1, 
        metadata={"help": "forwardChoose2 [0,1]"}
    )
    
    # model params
    alpha : float = field(
        default=0.1,
        metadata={"help": "alpha for loss"}
    )
    gradient_accumulation_steps: int = field(
        default = 1,
        metadata={"help": "gradient_accumulation_steps"}
    )
    enable_contrast:  bool = field(
        default= False,
        metadata={"help": "enable_contrast."}
    )
    enable_fsdp: bool = field(
        default= True,
        metadata={"help": "enable_fsdp."}
    )
    hidden_size: int = field(
        default=4096,
        metadata={"help": "coadellam-7b config.json hidden_size."}
    )
    
    vocab_size: int = field(
        default=32016,
        metadata={"help": "coadellam-7b config.json vocab_size."}
    )
    
    freezeLM: bool = field(
        default= True,
        metadata={"help": "freeze LM params."}
    )
    
    save_model: bool = field(
        default= True,
        metadata={"help": "save_model."}
    )
    
    model_name_or_path: str = field(
        default="../CodeLlama-7b-Instruct-hf", 
        metadata={"help": "the path to load pretrained model."}
    )

    tokenizer_path: str = field(
        default="../CodeLlama-7b-Instruct-hf", 
        metadata={"help": "the path to load pretrained tokenizer."}
    )
    pooling_type: str = field(
        default="average",
        metadata={"help": "the pooling method for reward model, selected from [average, max, last]."}
    )

    best_epoch: int = field(
        default=0,
        metadata={"help": "eval best epoch."}
    ) 
    
    # mode generate
    
    temperature: float = field( default=0.0, metadata={"help": "generate temperature"})
    top_p: float = field( default=0.9, metadata={"help": "generate top_p"})
    max_total_seq_len: int = field(default=3000,metadata={"help": "generate_length."})
    max_generate_length: int = field(default=1000,metadata={"help": "generate_length."})  
    # experiment setups
    
    output_dir: str = field(
        default="stylePrompt_model/stylePrompt_modelA", 
        metadata={"help": "output_dir"}
    )
    
    output_dir2: str = field(
        default="stylePrompt_model/stylePrompt_modelB", 
        metadata={"help": "output_dir2"}
    )
    
    predict_dirs: str = field(
        default="../out_predict/result_part.json", 
        metadata={"help": "predict_dirs"}
    )
    
    # tokenizer params
    padding_side: str = field(
        default="right",
        metadata={"help": "the direction for tokenizer to add padding tokens."}
    )
    

    # data params
    language : str = field(default="Java",metadata={"help": "language data."})   
    problem_path: str = field(
        default="../data/content_compelete.json",
        metadata={"help": "the path to load data."}
    )   
    human_eval_path: str = field(
        default="../data/humaneval_java.jsonl",
        metadata={"help": "humaneval_java datasets paths."}
    )
    human_eval_out_path: str = field(
        default="../out_predict/humaneval_java_out.jsonl",
        metadata={"help": "humaneval_java datasets paths."}
    )
    train_data_path: List[str] = field(
        default_factory=lambda: ["../data/Java_programming/Java_programming_train.json"],
        metadata={"help": "train datasets paths."}
    )

    eval_data_path: List[str] = field(
        default_factory=lambda: ["../data/Java_programming/Java_programming_dev.json"],
        metadata={"help": "evaluation datasets paths."}
    )

    test_data_path: List[str] = field(
        default_factory=lambda: ["/home/develop/dzl/PreferCodeLlama/data/Java_programming/Java_programming_test.json"],
        metadata={"help": "train datasets paths."}
    )
    
    feature_train_data_path: List[str] = field(
        default_factory=lambda: ["../data/Java_programming/Java_feature/code_styleFeature_train.json"],
        metadata={"help": "train datasets paths."}
    )
    
    feature_dev_data_path: List[str] = field(
        default_factory=lambda: ["../data/Java_programming/Java_feature/code_styleFeature_dev.json"],
        metadata={"help": "train datasets paths."}
    )
    
    user_style_data_path: str = field(
        default="../data/Java_programming/Java_feature/user_style.json",
        metadata={"help": "train datasets paths."}
    )
    
    
    # training hyperparams
    learning_rate2:float = field(default=1e-5, metadata={"help": "model stage 2 learining_rate"})
    num_feature_train_epochs: int = field(default= 30,metadata={"help": "num_feature_train_epochs."})
    
    per_device_feature_train_batch_size: int = field(default= 2,metadata={"help": "per_device_feature_train_batch_size."})
    
    per_device_feature_dev_batch_size: int = field(default= 2,metadata={"help": "per_device_feature_dev_batch_size."})
    
    per_device_test_batch_size: int = field(default= 2,metadata={"help": "per_device_test_batch_size."})
    
    do_train_first: bool = field(
        default=True,
        metadata={"help": "train first stage"}
    )
    do_train_second: bool = field(
        default=True,
        metadata={"help": "train second stage"}
    )
    
    eval_at_start: bool = field(
        default=False,
        metadata={"help": "whether make eval at start."}
    )

    debug_mode: bool = field(
        default=False,
        metadata={"help": "whether use the debug mode."}
    )

    weight_decay: float = field(default=0.0, metadata={"help": "weight_decay"})
    
    lm_loss_coeff: float = field(default=0., metadata={"help": "the coefficient for language modeling loss."})

    contrast_loss_coeff: float = field(default=0., metadata={"help": "the coefficient for contrastive learning loss."})

    gamma: float = field(default=0.85, metadata={"help": "model gamma"})

    max_length: int = field(
        default=4096,
        metadata={"help": "the max sentence sequence length."}
    )   



    

