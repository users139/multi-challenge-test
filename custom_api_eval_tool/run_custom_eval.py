import requests
import json
import os # For potential future use
from tqdm import tqdm
from collections import defaultdict # Added import
import csv # Added import for CSV saving

# --- Hardcoded Configurations ---
API_URL = "https://74.235.187.172:9443/callcenter/callBack/audio"
API_USERNAME = "llm-deepseek-hw-data-science"
API_PASSWORD = "T43RtmIfNcw6ZhzPuQSK7EJ9dGWyVAawx3pf"

HTTP_PROXY = "http://l50047843:%40LyfHuaWei139New7762@hkgpqwg00206.huawei.com:8080"
HTTPS_PROXY = "http://l50047843:%40LyfHuaWei139New7762@hkgpqwg00206.huawei.com:8080"
PROXIES = {"http": HTTP_PROXY, "https": HTTPS_PROXY}

GENERATION_MODEL_TYPE = "deepseek-v3"
GENERATION_PARAMS = {"temperature": 0.9, "top_p": 0.7, "max_tokens": 500}

EVALUATION_MODEL_TYPE = "deepseek-v3"
EVALUATION_PARAMS = {"temperature": 0.2, "top_p": 1.0, "max_tokens": 350}

# --- Data File Path ---
DATA_FILE_PATH = "../data/benchmark_questions.jsonl"
DETAILED_RESULTS_CSV_PATH = "custom_eval_detailed_results.csv" # Output in current dir

# --- Judge Prompt Template ---
JUDGE_PROMPT_TEMPLATE = """You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO". Your output MUST be a valid JSON object with keys "reasoning" and "verdict" (e.g., {"reasoning": "The model correctly answered...", "verdict": "YES"})."""


def call_custom_api(prompt_text: str, model_type: str, api_params: dict, is_evaluation_call: bool) -> str:
    # ... (call_custom_api function remains the same as in the previous step) ...
    payload = {
        "prompt": prompt_text,
        "model_type": model_type,
        "temperature": api_params.get("temperature"),
        "top_p": api_params.get("top_p"),
        "max_tokens": api_params.get("max_tokens")
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    try:
        response = requests.post(
            API_URL,
            json=payload,
            auth=(API_USERNAME, API_PASSWORD),
            proxies=PROXIES,
            verify=False,
            timeout=120
        )
        response.raise_for_status()

        if is_evaluation_call:
            return response.text
        else:
            try:
                data = response.json()
                if isinstance(data, dict):
                    if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                        choice = data["choices"][0]
                        if isinstance(choice, dict):
                            if "text" in choice:
                                return choice["text"]
                            if "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                                return choice["message"]["content"]
                    if "text" in data:
                        return data["text"]
                    if "generated_text" in data:
                        return data["generated_text"]
                    if "data" in data and isinstance(data["data"], str):
                         return data["data"]
                    return f"Error: Successfully called API but couldn't extract text from response: {json.dumps(data)}"
                return f"Error: Response JSON was not a dictionary: {json.dumps(data)}"
            except ValueError:
                return f"Error: Failed to decode JSON response. Raw response: {response.text}"
    except requests.exceptions.Timeout:
        return f"Error: API call timed out after {120} seconds."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP error occurred: {e}. Response: {e.response.text if e.response else 'No response body'}"
    except requests.exceptions.RequestException as e:
        return f"Error: API call failed due to a network issue: {str(e)}"
    except Exception as e:
        return f"Error: An unexpected error occurred: {str(e)}"

def save_detailed_results(results_list: list, filename: str):
    """Saves the detailed evaluation results to a CSV file."""
    if not results_list:
        print("Warning: No results to save.")
        return

    # Ensure all dictionaries have the same keys for CSV writing, handle potential missing keys gracefully.
    # Collect all possible keys from all dictionaries
    fieldnames_set = set()
    for item in results_list:
        fieldnames_set.update(item.keys())
    fieldnames = sorted(list(fieldnames_set)) # Sort for consistent column order

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore') # extrasaction='ignore' is safer
            writer.writeheader()
            for item in results_list:
                # Convert list/dict fields to JSON strings for CSV compatibility
                processed_item = {}
                for key, value in item.items():
                    if isinstance(value, (list, dict)):
                        try:
                            processed_item[key] = json.dumps(value)
                        except TypeError:
                            processed_item[key] = str(value) # Fallback for non-serializable complex types
                    else:
                        processed_item[key] = value
                writer.writerow(processed_item)
        print(f"Detailed results saved to {filename}")
    except IOError:
        print(f"Error: Could not write detailed results to {filename}. Check permissions or path.")
    except Exception as e:
        print(f"An unexpected error occurred while saving detailed results: {e}")


if __name__ == "__main__":
    print("Custom API Evaluation Tool - Standalone Runner")
    print(f"API URL: {API_URL}")
    print(f"Generation Model: {GENERATION_MODEL_TYPE}")
    print(f"Evaluation Model: {EVALUATION_MODEL_TYPE}")
    print(f"Loading data from: {DATA_FILE_PATH}")

    benchmark_data = []
    # ... (Data loading logic remains the same) ...
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    benchmark_data.append({
                        'QUESTION_ID': data.get('QUESTION_ID'),
                        'AXIS': data.get('AXIS'),
                        'CONVERSATION': data.get('CONVERSATION', []),
                        'TARGET_QUESTION': data.get('TARGET_QUESTION'),
                        'PASS_CRITERIA': data.get('PASS_CRITERIA')
                    })
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line due to JSON decode error: {e} - Line: {line.strip()}")
                except KeyError as e:
                    print(f"Warning: Skipping line due to missing key: {e} - Line: {json.loads(line)}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_FILE_PATH}.")
        exit(1)
    except Exception as e:
        print(f"Error loading data file: {e}")
        exit(1)
    if not benchmark_data:
        print("No benchmark data loaded. Exiting.")
        exit(1)
    print(f"Loaded {len(benchmark_data)} benchmark items.")

    # --- Response Generation Phase ---
    generation_results = []
    # ... (Response generation logic remains the same) ...
    print("\n--- Starting Response Generation ---")
    for item in tqdm(benchmark_data, desc="Generating Responses"):
        generation_prompt_text = "Error: Invalid conversation history."
        if not item.get('CONVERSATION') or not isinstance(item['CONVERSATION'], list):
            generated_response = "Error: Missing or invalid conversation history in input data."
        else:
            prompt_parts = [turn['content'] for turn in item['CONVERSATION'] if isinstance(turn, dict) and 'content' in turn]
            if not prompt_parts:
                 generated_response = "Error: No content found in conversation history to form a prompt."
            else:
                generation_prompt_text = "\n\n---\n\n".join(prompt_parts)
                generated_response = call_custom_api(
                    generation_prompt_text, GENERATION_MODEL_TYPE, GENERATION_PARAMS, is_evaluation_call=False
                )
        generation_results.append({
            'question_id': item['QUESTION_ID'], 'axis': item['AXIS'],
            'conversation_history': item['CONVERSATION'],
            'target_question_criteria': item['TARGET_QUESTION'],
            'expected_verdict': item['PASS_CRITERIA'],
            'generation_prompt': generation_prompt_text, 'generated_response': generated_response
        })
    print(f"\n--- Response Generation Complete ---")
    print(f"Generated {len(generation_results)} responses.")

    # --- Evaluation Phase ---
    evaluation_outputs = []
    # ... (Evaluation logic remains the same) ...
    print("\n--- Starting Evaluation ---")
    for result_item in tqdm(generation_results, desc="Evaluating Responses"):
        generated_response = result_item['generated_response']
        criteria_text = result_item['target_question_criteria']
        judge_prompt_text = "N/A - Skipped"; raw_eval_response = "N/A - Skipped"; judgement_reasoning = "N/A - Skipped"; judgement_verdict = "ERROR"
        if generated_response.startswith("Error:"):
            judgement_reasoning = f"Evaluation skipped: Response generation failed. Details: {generated_response}"; judgement_verdict = "NO"; raw_eval_response = generated_response
        elif not criteria_text:
            judgement_reasoning = "Evaluation skipped: Criteria (target_question) is missing."; judgement_verdict = "NO"
        else:
            judge_prompt_text = JUDGE_PROMPT_TEMPLATE.format(generated_response, criteria_text)
            raw_eval_response = call_custom_api(judge_prompt_text, EVALUATION_MODEL_TYPE, EVALUATION_PARAMS, is_evaluation_call=True)
            if raw_eval_response.startswith("Error:"):
                judgement_reasoning = f"Evaluation API call failed: {raw_eval_response}"; judgement_verdict = "NO"
            else:
                try:
                    eval_data = json.loads(raw_eval_response)
                    judgement_reasoning = eval_data.get("reasoning", "Error: Missing 'reasoning' in evaluation response.")
                    judgement_verdict = eval_data.get("verdict", "Error: Missing 'verdict' in evaluation response.")
                    if judgement_verdict not in ["YES", "NO"]:
                        judgement_reasoning += f" (Original verdict: '{judgement_verdict}' was invalid, defaulted to NO)"; judgement_verdict = "NO"
                except json.JSONDecodeError:
                    judgement_reasoning = f"Evaluation failed: Could not decode JSON response from evaluator. Raw: {raw_eval_response[:500]}..."; judgement_verdict = "NO"
                except Exception as e:
                    judgement_reasoning = f"Evaluation failed: Error parsing evaluation response: {str(e)}. Raw: {raw_eval_response[:500]}..."; judgement_verdict = "NO"
        passed_eval = (judgement_verdict == result_item['expected_verdict'])
        current_eval_output = result_item.copy()
        current_eval_output.update({
            'judge_prompt': judge_prompt_text, 'raw_evaluation_response': raw_eval_response,
            'judge_reasoning': judgement_reasoning, 'judge_verdict': judgement_verdict, 'passed_eval': passed_eval
        })
        evaluation_outputs.append(current_eval_output)
    print(f"\n--- Evaluation Complete ---")
    print(f"Completed {len(evaluation_outputs)} evaluations.")

    if evaluation_outputs: # Proceed only if there are results
        # --- Score Calculation ---
        axis_scores = defaultdict(lambda: {'passed': 0, 'total': 0})
        overall_passed = 0
        overall_total = 0

        for eval_item in evaluation_outputs:
            axis = eval_item.get('axis', 'Unknown Axis') # Handle if axis is somehow missing
            passed = eval_item.get('passed_eval', False) # Default to False if key is missing

            axis_scores[axis]['total'] += 1
            overall_total += 1
            if passed:
                axis_scores[axis]['passed'] += 1
                overall_passed += 1

        overall_score_percent = (overall_passed / overall_total * 100) if overall_total > 0 else 0
        axis_score_percent = {}
        for axis, counts in axis_scores.items():
            axis_score_percent[axis] = (counts['passed'] / counts['total'] * 100) if counts['total'] > 0 else 0

        # --- Reporting ---
        print("\n--- Evaluation Summary ---")
        print(f"Overall Score: {overall_score_percent:.2f}% ({overall_passed}/{overall_total} passed)")
        print("\nAxis Scores:")
        for axis, percent in sorted(axis_score_percent.items()): # Sort for consistent output
            counts = axis_scores[axis]
            print(f"- {axis}: {percent:.2f}% ({counts['passed']}/{counts['total']} passed)")

        # --- Save Detailed Results ---
        save_detailed_results(evaluation_outputs, DETAILED_RESULTS_CSV_PATH)
    else:
        print("\nNo evaluation outputs to process for scoring or saving.")

    print("\nStandalone script execution finished.")
