import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

start_time = time.perf_counter()

completion = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B:fastest",
    messages=[
        {
            "role": "user",
            "content": "Write a detailed essay about the history of artificial intelligence, its major milestones, and future implications. Include at least 500 words.",
        }
    ],
    max_tokens=1000,
)

end_time = time.perf_counter()

latency = (end_time - start_time) * 1000
tokens = completion.usage.completion_tokens

print(f"Response: {completion.choices[0].message.content}")
print(f"Latency: {latency:.2f} ms")
print(f"Tokens: {tokens}")
print(f"Throughput: {tokens / (latency / 1000):.2f} tokens/sec")
