🚀 Hackathon Submission Guidelines (OpenEnv RL Challenge)
1. Project Structure
Your inference script must be named inference.py
It must be located in the root directory of your project



2. LLM Usage Requirements
You must use the OpenAI Client for all LLM calls
Do not use alternative SDKs or direct HTTP calls



3. Required Environment Variables
Your inference.py must read the following environment variables:
API_BASE_URL
Description: API endpoint for the LLM
Requirement: Must include a default value
MODEL_NAME
Description: Model identifier used for inference
Requirement: Must include a default value
HF_TOKEN
Description: Hugging Face API token
Requirement: Mandatory (no default required)

4. INFERENCE OUTPUT FORMAT
The script must emit exactly three line types to stdout, in this order:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>



  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 rewards=0.00,0.00,1.00
✅ Example (inference.py)
import os
from openai import OpenAI

# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_inference(prompt: str):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    response = response.choices[0].message.content
    # Print output based on above given format


if __name__ == "__main__":
    print(run_inference("Hello from OpenEnv!"))

4. Hugging Face Space Guidelines
Building a Hugging Face Space can take significant time, especially if multiple spaces are active
To avoid delays:
Turn off all unnecessary spaces
Keep only your primary submission space running

5. Submission Validation Rules
The system will check if your Hugging Face Space is live
If your space is not in a running state, your submission will fail automatically
Before submitting:
Ensure your space is fully built
Confirm it is in the “Running” state

6. Hardware Requirements
Your solution will be executed inside a Docker container with limited resources
It must run within the following constraints:
2 vCPU
8 GB RAM
👉 Ensure your model, dependencies, and runtime fit within these limits. Submissions exceeding these constraints may fail during evaluation.

6. Resubmissions
You are allowed to resubmit your project multiple times
If your submission fails validation, you can:
Fix the issues
Ensure your Hugging Face Space is running
Submit again
👉 There is no penalty for resubmitting, so iterate until your submission passes all checks.

⚠️ Common Failure Cases (Avoid These)
inference.py not in root directory
Missing default values for API_BASE_URL or MODEL_NAME
Missing HF_TOKEN
Hugging Face Space still building during submission
Space stopped due to multiple active deployments

🚀  Reference projects to guide you
Here are some strong examples from the San Francisco edition to help you understand how to structure your environment:
Calendar Environment Server
https://github.com/meta-pytorch/OpenEnv/tree/main/envs/calendar_env
Reasoning Gym Environment Server
https://github.com/meta-pytorch/OpenEnv/tree/main/envs/reasoning_gym_env
TB2 Environment Server
https://github.com/meta-pytorch/OpenEnv/tree/main/envs/tbench2_env
CARLA Environment Server
https://github.com/meta-pytorch/OpenEnv/tree/main/envs/carla_env
REPL Environment Server
https://github.com/meta-pytorch/OpenEnv/tree/main/envs/repl_env
Use these as direction, not as templates. Focus on understanding structure and approach.

