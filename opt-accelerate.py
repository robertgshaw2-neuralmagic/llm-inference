import gc, io, json, math, os, time
from argparse import ArgumentParser
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import time, gc, torch
from tqdm import tqdm

num_tokens = 100
BATCH_SIZES = [1,4,8,16,32,64,128]

parser = ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="model_name", default="facebook/opt-2.7b")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
args = parser.parse_args()

model_name = args.name
dtype = torch.float16

print(f"*** Loading the model {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding="left")
kwargs = dict(
    device_map="balanced_low_0",
    torch_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
model = model.eval()


input_arr = ["In the far far distance from our galaxy"]
generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)
print(f"Generate args {generate_kwargs}")

def generate(batch_sz=1):
    inputs = input_arr * batch_sz
    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda:0")

    output_tokens = model.generate(**input_tokens, **generate_kwargs)
    outputs = tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    return output_tokens, outputs


print("*** Running generate warmup")
output_tokens, outputs = generate(batch_sz=BATCH_SIZES[0])
print(f"output_tokens.shape = {output_tokens.shape}")
print(f"outputs[0] = {outputs[0]}")
print(f"outputs[-1] = {outputs[-1]}")

if args.benchmark:
    torch.cuda.empty_cache()
    gc.collect()

# benchmark it!
if args.benchmark:
    for batch_sz in BATCH_SIZES:
        print(f"*** Running benchmark with batch_sz = {batch_sz}")

        # warm up
        for i in range(1):
            _ = generate(batch_sz=batch_sz)
        torch.cuda.synchronize()

        # benchmark
        t0 = time.time()
        cycles = 3
        total_tokens = 0
        for i in tqdm(range(cycles)):
            output_tokens, _ = generate(batch_sz=batch_sz)
            total_tokens += output_tokens.shape[0] * output_tokens.shape[1]
            
        torch.cuda.synchronize()
        t1 = time.time()
        
        throughput = total_tokens / (t1 - t0)
        
        print(
            f"""
    *** Performance stats for batch = {batch_sz}:
    Throughput: {throughput:.2f} tokens/sec
    Total Tokens: {total_tokens} tokens
    Total Time: {t1 - t0:.2f} secs
    """
        )
        
        torch.cuda.empty_cache()
        gc.collect()
