import os
import time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load your Hugging Face token from the .env file
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("Please set your HF_TOKEN in the .env file.")

# Initialize the official Hugging Face Inference Client
client = InferenceClient(api_key=HF_TOKEN)

# List of models and their specific providers to test.
MODELS_TO_TEST = [
    # Testing Cerebras hardware (Ultra-fast Meta models)
    {"model": "meta-llama/Llama-3.3-70B-Instruct", "provider": "cerebras"},
    
    # Testing DeepSeek using the :fastest routing policy!
    {"model": "deepseek-ai/DeepSeek-R1", "provider": "fastest"},
    
    # Testing Together AI
    {"model": "meta-llama/Llama-3.2-1B-Instruct", "provider": "together"},
    
    # Auto-routing (no suffix)
    {"model": "Qwen/Qwen2.5-7B-Instruct", "provider": None}
]

PROMPT = "Explain the theory of relativity in one simple paragraph."
MAX_TOKENS = 100
NUM_RUNS = 3

def test_model_speed(model_config: dict) -> dict:
    base_model = model_config["model"]
    provider = model_config["provider"]
    
    # THE FIX: Construct the model string with the :provider suffix
    target_model = f"{base_model}:{provider}" if provider else base_model
    
    print(f"\nTesting: {target_model}...")
    
    latencies = []
    throughputs = []
    
    for i in range(NUM_RUNS):
        start_time = time.perf_counter()
        
        try:
            # We removed the provider kwarg and pass the suffixed string directly
            response = client.chat_completion(
                model=target_model,
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=MAX_TOKENS,
                stream=False
            )
            
            end_time = time.perf_counter()
            time_taken_sec = end_time - start_time
            latency_ms = time_taken_sec * 1000
            latencies.append(latency_ms)
            
            completion_tokens = response.usage.completion_tokens
            
            tps = completion_tokens / time_taken_sec if time_taken_sec > 0 else 0
            if completion_tokens > 0:
                throughputs.append(tps)
            
            print(f"  Run {i+1}: {latency_ms:.2f} ms | {tps:.2f} tokens/sec ({completion_tokens} tokens)")
            
        except Exception as e:
            error_msg = str(e).split('\n')[0]
            print(f"  Run {i+1}: Error - {error_msg}")
            
            if "not supported" in error_msg.lower() or "not found" in error_msg.lower() or "404" in error_msg:
                print(f"  -> Skipping remaining runs for {target_model}")
                break

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        avg_tps = sum(throughputs) / len(throughputs) if throughputs else 0
        
        return {
            "display_name": target_model,
            "avg_latency_ms": avg_latency,
            "avg_tps": avg_tps
        }
    return None

def main():
    print("Starting HF InferenceClient Speed Test (Suffix Method)...")
    print(f"Prompt: '{PROMPT}'")
    print("-" * 80)
    
    results = []
    
    for config in MODELS_TO_TEST:
        result = test_model_speed(config)
        if result:
            results.append(result)
            
    print("\n\n" + "=" * 80)
    print(f"{'MODEL:PROVIDER':<45} | {'AVG LATENCY (ms)':<18} | {'AVG TOKENS/SEC':<15}")
    print("=" * 80)
    
    if not results:
        print("No successful runs to report.")
        return
        
    results.sort(key=lambda x: x["avg_tps"], reverse=True)
    
    for r in results:
        print(f"{r['display_name']:<45} | {r['avg_latency_ms']:<18.2f} | {r['avg_tps']:.2f}")

if __name__ == "__main__":
    main()