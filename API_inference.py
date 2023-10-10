from transformers import LlamaTokenizer, LlamaForCausalLM,GenerationConfig
import torch
from typing import List, Literal, Optional, Tuple, TypedDict
PATH_TO_CONVERTED_WEIGHTS = "./CodeLlama-7b-Instruct-hf"
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."
Role = Literal["system", "user", "assistant"]
class Message(TypedDict):
    role: Role
    content: str
Dialog = List[Message]

model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
model = model.cuda()
tokenizer = LlamaTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

         
    


def generate_prompt(dialogs: List[Dialog]):
    prompt_tokens = []
    unsafe_requests = []
    for dialog in dialogs:

        unsafe_requests.append(
            any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
        )
        if dialog[0]["role"] == "system":
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
        assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
            [msg["role"] == "assistant" for msg in dialog[1::2]]
        ), (
            "model only supports 'system', 'user' and 'assistant' roles, "
            "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
        )
        dialog_tokens: List[int] = sum(
            [
                tokenizer.encode(
                    f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                    bos=True,
                    eos=True,
                )
                for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
            ],
            [],
        )
        assert (
            dialog[-1]["role"] == "user"
        ), f"Last message must be from user, got {dialog[-1]['role']}"
        dialog_tokens += tokenizer.encode(
            f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
            bos=True,
            eos=False,
        )
        prompt_tokens.append(dialog_tokens)
    return prompt_tokens
Len = 0
def evaluate(
            instruction,
            input=None,
            temperature=0.2,
            top_p=0.95,
            top_k=40,
            num_beams=4,
            max_new_tokens=1024,
            **kwargs,
    ):

        #prompt = generate_prompt(instruction)
        #print(prompt)
        inputs = tokenizer(instruction, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        #input_ids = torch.tensor(prompt).cuda()
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )

        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print(output.split(E_INST)[1])
        return output

if __name__ == "__main__":
    #instruction = "a function that computes the set of sums of all contiguous sublists of a given list."
    #instruction =B_SYS + "Provide answers in C++"+ E_SYS + instruction
    #instruction = f"{B_INST} {(instruction).strip()} {E_INST}"
    instruction = "<s>[INST] <<SYS>>\nProvide answers in Java\n<</SYS>>\n\nUse exception handling input mechanism to make the program more robust.\n\n## main method:\n1. Input n and create an int array of size n.\n2. Input n integers and put them into the array. When inputting, it is possible to input non-integer strings. In this case, output the exception information and then re-enter.\n3. Use `Arrays.toString` to output the contents of the array.\n\n## Input example:\n```in\n5\n1\n2\na\nb\n4\n5\n3\n\n```\n## Output example:\n```out\njava.lang.NumberFormatException: For input string: \"a\"\njava.lang.NumberFormatException: For input string: \"b\"\n[1, 2, 4, 5, 3]\n\n``` [/INST]"
    Len = len(instruction)
    print(instruction)
    print("---------------------")
    evaluate(instruction = instruction)