# MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs
MultiChallenge is a novel benchmark designed to evaluate large language models (LLMs) on their ability to handle multi-turn conversations with human users—an essential but underexplored capability for their real-world applications. MultiChallenge focuses on four key categories of challenges that are common, realistic, and highly demanding in current human-LLM interactions. These challenges require LLMs to excel simultaneously in accurate context allocation, in-context reasoning, and instruction-following.
## **Project Structure**
- `data/` : Contains input files for conversations (benchmark_questions.jsonl) and optional model response files (in final_model_responses) used in the benchmark.
- `results/` : Stores the benchmark's output, including evaluation scores and metrics, saved to evaluation_results.txt.
- `src/` : Core functionality for the benchmark
  - `models/` : Houses model provider classes:
  
## **Setup Instructions**
1. **Clone the Repository**
   ```bash
   git clone some_directory
   cd multi-challenge
   ```
2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```
3. **Create `.env` File**
   Create a `.env` file in the root directory with your API keys. For example:
   ```plaintext
   OPENAI_API_KEY=your-openai-api-key (REQUIRED)
   HUGGINGFACE_TOKEN=your-huggingface-token
   ```
## **Usage**
### **1. Using Pre-Generated Responses**
If you already have model responses and want to evaluate them:
```bash
python main.py --responses-file data/model_responses.jsonl --output-file results/evaluation_results.txt
```
Make sure to format the responses file as shown in data/responses_template.jsonl

### **2. Generating Responses with a Model**
To dynamically generate responses using a supported model provider:
```bash
python main.py --model-provider openai --provider-args model=gpt-4o temp=0 --output-file results/evaluation_results.txt
```

### **3. Using Multiple Attempts**
To evaluate model performance with multiple attempts per conversation:
```bash
python main.py --model-provider openai --attempts 3 --output-file results/evaluation_results.txt
```
This will generate 3 responses per conversation and consider it successful if any attempt passes.

### **4. Generating Detailed Raw Output**
To save comprehensive evaluation details including all responses and judgments:
```bash
python main.py --model-provider openai --attempts 3 --output-file results/evaluation_results.txt --raw results/detailed_results.csv
```

### **Command-Line Arguments**
- `--output-file`: Path to save the final evaluation results.
- `--responses-file`: Path to a file containing pre-generated responses. (OPTIONAL)
- `--model-provider`: Specify the model provider for generating responses (e.g., `huggingface`, `openai`, `custom_requests`).
- `--provider-args`: Model-specific arguments in `key=value` format (e.g., `model_path=/path/to/model`). For `custom_requests`, see the "Testing Your Own API" section for required and optional arguments.
- `--evaluator_model_provider`: Specify the model provider for evaluation (e.g., `openai`, `custom_requests`). Defaults to `openai` using GPT-4o if not specified.
- `--evaluator_provider_args`: Provider-specific arguments for the evaluator model in `key=value` format. Similar to `--provider-args`. For `custom_requests` details, see 'Testing Your Own API' section.
- `--attempts`: Number of attempts to generate for each conversation. Defaults to 1. 
- `--max-workers_response_gen`: Number of concurrent workers to multi-thread response generation. Defaults to 1.
- `--max-workers_eval`: Number of concurrent workers to multi-thread response evaluation. Defaults to 1.
- `--raw`: Path to save detailed raw output including all responses and evaluations. (OPTIONAL)

### **Evaluation Results**
The evaluation results include:
1. **In evaluation_results.txt:**
   - Overall Score: Percentage of conversations where at least one attempt meets the criteria
   - Axis Scores: Per-axis scores based on number of attempts

2. **In detailed_results.txt (if --raw is specified):**
   - Complete conversation history
   - All model responses for each attempt
   - Judge's verdicts and reasoning
   - Expected pass criteria
   - Per-conversation pass/fail statistics
---

## Testing Your Own API

This benchmark can be configured to use your own custom API for response generation, evaluation, or both. This is facilitated by the `custom_requests` model provider.

### Common Configuration for `custom_requests`

When using `--model-provider custom_requests` or `--evaluator_model_provider custom_requests`, you'll use `--provider-args` or `--evaluator_provider_args` respectively to pass necessary details for your API. These arguments are key-value pairs:

*   `url=<YOUR_API_ENDPOINT>`: **(Required)** The full URL of your API endpoint.
*   `model_id=<YOUR_MODEL_IDENTIFIER>`: **(Required)** A name or identifier for the model served by your API. This is included in the request payload.
*   `username=<YOUR_API_USERNAME>`: **(Required)** Username for basic authentication.
*   `password=<YOUR_API_PASSWORD>`: **(Required)** Password for basic authentication.
*   `temperature=<float>`: (Optional, default: 0.7) Temperature setting for generation. The value passed here will be automatically converted from string to float.
*   `top_p=<float>`: (Optional, default: 1.0) Top_p setting for generation. The value passed here will be automatically converted from string to float.
*   `max_tokens=<int_or_None>`: (Optional, default: None) Maximum tokens to generate. Use "None" (case-insensitive string) or an empty string for no limit, otherwise provide an integer. This will be automatically converted.
*   `http_proxy=<PROXY_URL>`: (Optional) URL for an HTTP proxy.
*   `https_proxy=<PROXY_URL>`: (Optional) URL for an HTTPS proxy.

**API Behavior Notes:**

*   **Authentication:** Currently, only HTTP Basic Authentication is supported by `custom_requests`.
*   **Request Format:** The `custom_requests` provider sends a POST request with a JSON payload like:
    ```json
    {
        "prompt": "Your processed prompt content here...",
        "model_id": "your_model_id",
        "temperature": 0.75,
        "top_p": 0.7,
        "max_tokens": 1000 // Only if not None
    }
    ```
*   **SSL Verification:** `verify=False` is used by default for requests made by `custom_requests`, so your API endpoint does not strictly need a publicly trusted SSL certificate for basic operation with this tool.

### 1. Using Your Custom API for Response Generation (as Model Under Test)

To test your API's ability to generate responses to the benchmark questions:

*   Set `--model-provider custom_requests`.
*   Provide your API details using `--provider-args` as described in "Common Configuration".
*   The API should expect a prompt string (which could be a simple string or a concatenation of chat history turns) and return a JSON response from which a textual answer can be extracted. The `custom_requests` model attempts to find text in common locations like `response.json()['choices'][0]['text']` or `response.json()['text']`.

**Example:**

```bash
python main.py \
    --model-provider custom_requests \
    --provider-args \
        url=https://your-api.example.com/generate \
        model_id=my-custom-model-v1 \
        username=myuser \
        password=mypassword \
        temperature=0.8 \
        max_tokens=1024 \
    --output-file results/custom_api_generation_results.txt \
    --raw results/custom_api_generation_raw.csv
```

### 2. Using Your Custom API for Evaluation

To use your API as the evaluator (judge) for responses generated by another model (or pre-loaded responses):

*   Set `--evaluator_model_provider custom_requests`.
*   Provide your API details using `--evaluator_provider_args` as described in "Common Configuration".
*   Add `evaluation_mode=True` to your `--evaluator_provider_args`. This tells the `custom_requests` model to return the raw JSON string from your API.
*   **Critical Requirement:** Your API **must** be designed to act as a "judge". It will receive a prompt formatted with the model's response and the evaluation criteria (see `JUDGE_PROMPT` in `src/evaluator.py`).
*   Your API **must** return a JSON string that strictly conforms to the `JudgeResponse` Pydantic model:
    ```json
    {
        "reasoning": "Detailed explanation of why the verdict was reached...",
        "verdict": "YES"
    }
    ```
    (The `verdict` must be either "YES" or "NO").
    The `custom_requests` model, when `evaluation_mode=True` is set in its arguments, will return this JSON string directly. The `Evaluator` will then parse this string to a `JudgeResponse` object.

**Example (evaluating pre-generated responses with your custom judge):**

```bash
python main.py \
    --responses-file data/model_responses.jsonl \
    --evaluator_model_provider custom_requests \
    --evaluator_provider_args \
        url=https://your-judge-api.example.com/evaluate \
        model_id=my-custom-judge-v1 \
        username=judgeuser \
        password=judgepass \
        temperature=0.1 \
        evaluation_mode=True \
    --output-file results/custom_judge_results.txt \
    --raw results/custom_judge_raw.csv
```

### 3. Using Your Custom API for Both Generation and Evaluation

You can use one custom API for generation and another (or the same, if it supports both modes and different prompt structures) for evaluation.

**Example:**

```bash
python main.py \
    --model-provider custom_requests \
    --provider-args \
        url=https://your-generation-api.example.com/generate \
        model_id=my-generation-model \
        username=gen_user \
        password=gen_pass \
        temperature=0.9 \
    --evaluator_model_provider custom_requests \
    --evaluator_provider_args \
        url=https://your-evaluation-api.example.com/evaluate \
        model_id=my-evaluation-model \
        username=eval_user \
        password=eval_pass \
        temperature=0.2 \
        evaluation_mode=True \
    --output-file results/full_custom_pipeline_results.txt \
    --raw results/full_custom_pipeline_raw.csv
```

## **Project Dependencies**
See `requirements.txt` for a complete list of required packages.
---
