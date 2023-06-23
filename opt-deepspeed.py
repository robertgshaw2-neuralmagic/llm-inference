import gc, io, json, math, os, time
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.distributed as dist

import deepspeed
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.bloom.modeling_bloom import BloomBlock as BloomBlock
from transformers.utils import is_offline_mode

t_start = time.time()

num_tokens = 100
BATCH_SIZES = [64, 32, 16, 8, 4, 1]

parser = ArgumentParser()

parser.add_argument("--name", required=True, type=str, help="model_name", default="facebook/opt-2.7b")
parser.add_argument("--local_rank", required=False, type=int, help="used by dist launchers")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
args = parser.parse_args()

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))

deepspeed.init_distributed("nccl")
rank = dist.get_rank()

def print_rank0(*msg):
    if rank != 0:
        return
    print(*msg)

model_name = args.name
dtype = torch.float16

print_rank0(f"*** Loading the model {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name, padding="left")

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("pre-from-pretrained", force=True)


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
model = model.eval()
print_rank0(f"*** Done loading the model")

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("post-init-ds-zero-init", force=True)

# ## Deepspeed-Inference Loading

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("pre-ds-inference-init", force=True)

model = deepspeed.init_inference(
    model,
    mp_size=world_size,
    dtype=dtype,
    replace_with_kernel_inject=True,
)

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("post-ds-inference-init", force=True)


if args.benchmark:
    t_ready = time.time()


# ## Generate

input_arr = ["In the far far distance from our galaxy"]

generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)

print_rank0(f"Generate args {generate_kwargs}")

def generate(batch_sz=1):
    inputs = input_arr * batch_sz
    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to(torch.cuda.current_device())

    output_tokens = model.generate(**input_tokens, **generate_kwargs)
    outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    return output_tokens, outputs


print_rank0("*** Running generate warmup")
output_tokens, outputs = generate(batch_sz=BATCH_SIZES[0])
print_rank0(f"output_tokens.shape = {output_tokens.shape}")
print_rank0(f"outputs[0] = {outputs[0]}")
print_rank0(f"outputs[-1] = {outputs[-1]}")

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()
    deepspeed.runtime.utils.see_memory_usage("end-of-run", force=True)

# ## Benchmark

# benchmark it!
if args.benchmark:
    for batch_sz in BATCH_SIZES:
        print_rank0(f"*** Running benchmark with batch_sz = {batch_sz}")

        # warm up
        for i in range(1):
            _ = generate(batch_sz=batch_sz)
        torch.cuda.synchronize()

        # benchmark
        t0 = time.time()
        cycles = 3
        total_tokens = 0
        for i in range(cycles):
            output_tokens, _ = generate(batch_sz=batch_sz)
            total_tokens += output_tokens.shape[0] * output_tokens.shape[1]
            
        torch.cuda.synchronize()
        t1 = time.time()
        
        throughput = total_tokens / (t1 - t0)
        
        print_rank0(
            f"""
    *** Performance stats for batch = {batch_sz}:
    Throughput: {throughput:.2f} tokens/sec
    Total Tokens: {total_tokens} tokens
    Total Time: {t1 - t0:.2f} secs
    """
        )
        
        torch.cuda.empty_cache()
        gc.collect()
