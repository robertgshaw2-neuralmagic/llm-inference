import gc, io, json, math, os, time
from argparse import ArgumentParser
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import time, gc, torch
from tqdm import tqdm


BATCH_SIZES = [1,4,8,16,32,64,128]

parser = ArgumentParser()
parser.add_argument("--name", required=True, type=str, help="model_name", default="facebook/opt-2.7b")
parser.add_argument("--batch_size", default=1, type=int, help="batch size")
parser.add_argument("--cycles", default=3, type=int, help="number of cycles for testing")
parser.add_argument("--num_tokens", default=100, type=int, help="number of tokens to generate")
parser.add_argument("--benchmark", action="store_true", help="additionally run benchmark")
parser.add_argument("--do_long_prompt", action="store_true", help="do the long prompt")
args = parser.parse_args()

num_tokens = args.num_tokens
model_name = args.name
cycles = args.cycles
dtype = torch.float16

print(f"*** Loading the model {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding="left")
kwargs = dict(
    device_map="balanced_low_0",
    torch_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
model = model.eval()

if args.do_long_prompt:
    text = """###Prompt
Extract entities from the following sentences without changing original words.

###
Sentence: "5+ years of experience delivering scalable and resilient services at large enterprise scale, including experience in data platforms including large-scale analytics on relational, structured and unstructured data. 3+ years of experience as a SWE/Dev/Technical lead in an agile environment including 1+ years of experience operating in a DevOps model. 2+ years of experience designing secure, scalable and cost-efficient PaaS services onthe Microsoft Azure (or similar) platform."
DIPLOMA: none
DIPLOMA_MAJOR: none
EXPERIENCE: 3+ years, 5+ years, 5+ years, 5+ years, 3+ years, 1+ years, 2+ years
SKILLS: designing, delivering scalable and resilient services, data platforms, large-scale analytics on relational, structured and unstructured data, SWE/Dev/Technical, DevOps, designing, PaaS services, Microsoft Azure
###

Sentence: "8+ years demonstrated experience in designing and developing enterprise-level scale services/solutions. 3+ years of leadership and people management experience."
DIPLOMA: Bachelors
DIPLOMA_MAJOR: Computer Science
EXPERIENCE: 8+ years, 3+ years, 5+ years, 5+ years, 5+ years, 3+ years
SKILLS: designing, developing enterprise-level scale services/solutions, leadership and people management experience, Agile Experience, full-stack software development, C#, designing
"""
else:
    text = "In the far far distance from our galaxy"

input_arr = [text]
generate_kwargs = dict(max_new_tokens=num_tokens, do_sample=False)
print(f"Generate args {generate_kwargs}")

def generate(batch_sz=1):
    inputs = input_arr * batch_sz
    input_tokens = tokenizer.batch_encode_plus(inputs, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to("cuda:0")

    output_tokens = model.generate(**input_tokens, **generate_kwargs)
    print(output_tokens.shape)
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
with torch.no_grad():
    if args.benchmark:
        for batch_sz in BATCH_SIZES:
            print(f"*** Running benchmark with batch_sz = {batch_sz}, num_tokens = {num_tokens}")

            # warm up
            for i in range(1):
                _ = generate(batch_sz=batch_sz)
            torch.cuda.synchronize()

            # benchmark
            t0 = time.time()
            total_tokens = 0
            for i in tqdm(range(cycles)):
                output_tokens, _ = generate(batch_sz=batch_sz)
                total_tokens += output_tokens.shape[0] * output_tokens.shape[1]

            torch.cuda.synchronize()
            t1 = time.time()

            throughput = total_tokens / (t1 - t0)

            print(
                f"""
        *** Performance stats for batch = {batch_sz}, num_tokens = {num_tokens}:
        Throughput: {throughput:.2f} tokens/sec
        Total Tokens: {total_tokens} tokens
        Total Time: {t1 - t0:.2f} secs
        """
            )

            torch.cuda.empty_cache()
            gc.collect()