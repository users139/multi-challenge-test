import concurrent.futures
from tqdm import tqdm
from src.models.factory import ModelFactory # Assuming factory.py is in src/models/

# 1. Define the api_config dictionary
api_config = {
    # "provider_type": "custom_requests", # provider_type is used by the factory, not the model itself
    "url": "https://74.235.187.172:9443/callcenter/callBack/audio",
    "model_id": "deepseek-r1",
    "username": "llm-deepseek-hw-data-science",
    "password": "T43RtmIfNcw6ZhzPuQSK7EJ9dGWyVAawx3pf",
    "temperature": 0.75,
    "max_tokens": None,  # Or some integer value if desired, e.g., 150
    "top_p": 0.7,
    "http_proxy": "http://l50047843:%40LyfHuaWei139New7762@hkgpqwg00206.huawei.com:8080",
    "https_proxy": "http://l50047843:%40LyfHuaWei139New7762@hkgpqwg00206.huawei.com:8080",
}

# 2. Create a list of sample prompts
prompts_list = [
    "Hello, how are you today?",
    "What is the capital of France?",
    "Translate the word 'book' to Spanish.",
    "Explain the concept of black holes in simple terms.",
    "Write a short poem about the rain."
]

# 3. Define a function test_api_prompt(prompt)
def test_api_prompt(prompt: str):
    """
    Tests a single prompt against the custom API.
    """
    try:
        # Get the model provider instance using the factory
        # The factory will use 'custom_requests' to find the right class
        # and pass the rest of api_config to its __init__
        model = ModelFactory.get_provider('custom_requests', **api_config)

        print(f"\nSending prompt: '{prompt}'")
        response = model.generate(prompt)
        print(f"Received response for '{prompt}': '{response}'")
        return response
    except Exception as e:
        print(f"Error processing prompt '{prompt}': {e}")
        return None

# 4. Use ThreadPoolExecutor to run tests concurrently
if __name__ == "__main__":
    print("Starting API tests...\n")
    num_worker_threads = 5  # Reasonable default

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_worker_threads) as executor:
        # Wrap executor.map with tqdm for a progress bar
        # The list() call is to ensure all tasks are executed and tqdm can track completion
        future_to_prompt = {executor.submit(test_api_prompt, prompt): prompt for prompt in prompts_list}

        for future in tqdm(concurrent.futures.as_completed(future_to_prompt), total=len(prompts_list), desc="Testing Prompts"):
            prompt = future_to_prompt[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                print(f"Prompt '{prompt}' generated an exception: {exc}")

    print("\nAPI tests completed.")
    print(f"Successfully received {len(results)} responses out of {len(prompts_list)} prompts.")

    # You can further process or assert results here if needed
    # For example, check if all prompts returned non-None responses
    if len(results) == len(prompts_list):
        print("All prompts received a response.")
    else:
        print(f"{len(prompts_list) - len(results)} prompts failed to get a response.")
