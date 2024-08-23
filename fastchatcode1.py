# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
import torch
from torch import bfloat16

# Load LLM with less GPU memory
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)


tokenizer = AutoTokenizer.from_pretrained("lmsys/fastchat-t5-3b-v1.0")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "lmsys/fastchat-t5-3b-v1.0",
    quantization_config=bnb_config,
    device_map='auto',
)

from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline


pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=256
)

local_llm = HuggingFacePipeline(pipeline=pipe)


output = pipe("Ask prompt here.")
print(output)