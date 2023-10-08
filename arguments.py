from typing import List, Optional

from dataclasses import dataclass, field
from transformers import TrainingArguments

@dataclass
class train_config(TrainingArguments):
    # model params
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
        default="./CodeLlama-7b-Instruct-hf", 
        metadata={"help": "the path to load pretrained model."}
    )

    tokenizer_path: str = field(
        default="./CodeLlama-7b-Instruct-hf", 
        metadata={"help": "the path to load pretrained tokenizer."}
    )
    pooling_type: str = field(
        default="average",
        metadata={"help": "the pooling method for reward model, selected from [average, max, last]."}
    )

    best_epoch: int = field(
        default=1,
        metadata={"help": "eval best epoch."}
    ) 
    
    # experiment setups
    
    output_dir: str = field(
        default="normal", 
        metadata={"help": "output_dir"}
    )
    
    reward_domain: str = field(
        default="normal", 
        metadata={"help": "the domain for reward model training."}
    )
    
    # tokenizer params
    padding_side: str = field(
        default="right",
        metadata={"help": "the direction for tokenizer to add padding tokens."}
    )


    # data params

    problem_path: str = field(
        default="./data/content_compelete.json",
        metadata={"help": "the path to load data."}
    )   

    train_data_path: List[str] = field(
        default_factory=lambda: ["./data/Java_programming/train/Java_programming_train.json"],
        metadata={"help": "train datasets paths."}
    )


    eval_data_path: List[str] = field(
        default_factory=lambda: ["./data/Java_programming/dev/Java_programming_dev.json"],
        metadata={"help": "evaluation datasets paths."}
    )


    # training hyperparams
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



    

