from collections import namedtuple
from transformers import LlamaConfig, LlamaTokenizer, OPTConfig, Qwen2Config, Qwen2Tokenizer, GPT2Tokenizer
from src.catpllm.model.llm import LlamaModel, OPTModel, Qwen2Model

                        
LLMClass = namedtuple("LLMClass", ('config', 'tokenizer', 'model',))


_LLM_CLASSES = {
    "llama2": LLMClass(**{
        "config": LlamaConfig,
        "tokenizer": LlamaTokenizer,
        "model": LlamaModel
    }),
    "opt": LLMClass(**{
        "config": OPTConfig,
        "tokenizer": GPT2Tokenizer,
        "model": OPTModel
    }),
    'qwen2': LLMClass(**{
        "config": Qwen2Config,
        "tokenizer": Qwen2Tokenizer,
        "model": Qwen2Model
    })
}


def get_model_class(llm_type: str):
    if 'llama2' in llm_type or 'tinyllama' in llm_type:
        return _LLM_CLASSES['llama2']
    if 'opt' in llm_type:
        return _LLM_CLASSES['opt']
    if 'gpt2' in llm_type:
        return _LLM_CLASSES['gpt2']
    return None


def load_llm(llm_name, llm_path,  **kwargs):
    r"""A llm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        llm_name: name of the llm
        llm_path: path of the llm

    Returns:
        :obj:`PreTrainedModel`: The pretrained llm model.
        :obj:`tokenizer`: The llm tokenizer.
        :obj:`llm_config`: The config of the pretrained llm model.
    """
    llm_class = get_model_class(llm_type=llm_name)
    llm_config = llm_class.config.from_pretrained(llm_path)
    
    llm = llm_class.model.from_pretrained(llm_path, config=llm_config)
    tokenizer = llm_class.tokenizer.from_pretrained(llm_path) 

    return llm, tokenizer, llm_config