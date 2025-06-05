import requests
from requests.auth import HTTPBasicAuth
from src.models.base import ModelProvider  # Assuming this path is correct
from typing import Any, List, Dict, Optional # For type hinting

class CustomRequestsModel(ModelProvider):
    def __init__(self, url: str, model_id: str, username: str, password: str,
                 temperature: Any = 0.7,  # Default to a float, but expect string from factory
                 max_tokens: Any = None,   # Default to None, expect string or None
                 top_p: Any = 1.0,         # Default to a float, but expect string from factory
                 http_proxy: Optional[str] = None,
                 https_proxy: Optional[str] = None,
                 evaluation_mode: Any = False # Default to bool, expect string from factory
                 ):
        self.url = url
        self.model_id = model_id
        self.username = username
        self.password = password

        # Type conversion for temperature
        try:
            self.temperature = float(temperature)
        except ValueError:
            raise ValueError(f"Invalid temperature value: '{temperature}'. Must be a float.")

        # Type conversion for max_tokens
        if isinstance(max_tokens, str):
            if max_tokens.lower() == 'none' or max_tokens == '':
                self.max_tokens = None
            else:
                try:
                    self.max_tokens = int(max_tokens)
                except ValueError:
                    raise ValueError(f"Invalid max_tokens value: '{max_tokens}'. Must be an integer or 'None'.")
        elif max_tokens is None:
            self.max_tokens = None
        else: # Already an int or other non-string, non-None type
             try:
                self.max_tokens = int(max_tokens)
             except (ValueError, TypeError):
                raise ValueError(f"Invalid max_tokens value: '{max_tokens}'. Must be an integer, string 'None', or None.")


        # Type conversion for top_p
        try:
            self.top_p = float(top_p)
        except ValueError:
            raise ValueError(f"Invalid top_p value: '{top_p}'. Must be a float.")

        self.http_proxy = http_proxy
        self.https_proxy = https_proxy

        # Type conversion for evaluation_mode
        if isinstance(evaluation_mode, str):
            if evaluation_mode.lower() == 'true':
                self.evaluation_mode = True
            elif evaluation_mode.lower() == 'false':
                self.evaluation_mode = False
            else:
                raise ValueError(f"Invalid evaluation_mode value: '{evaluation_mode}'. Must be 'true' or 'false'.")
        elif isinstance(evaluation_mode, bool):
            self.evaluation_mode = evaluation_mode
        else:
            raise ValueError(f"Invalid evaluation_mode type: {type(evaluation_mode)}. Must be a string or boolean.")

    def generate(self, prompt: Any) -> str: # Changed prompt type to Any
        headers = {
            "Content-Type": "application/json"
        }

        actual_prompt_text: str
        if isinstance(prompt, str):
            actual_prompt_text = prompt
        elif isinstance(prompt, list):
            # Assuming prompt is a list of dictionaries (chat history)
            # Each dictionary is expected to have a 'content' key
            try:
                actual_prompt_text = "\n".join([turn['content'] for turn in prompt if 'content' in turn])
            except TypeError: # Handles cases where prompt might be list of non-dict items
                raise ValueError("If prompt is a list, it must be a list of dictionaries with a 'content' key.")
            except KeyError: # Should be caught by 'if content in turn' but as a safeguard
                raise ValueError("Each dictionary in the prompt list must contain a 'content' key.")
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}. Prompt must be a string or a list of dictionaries.")

        payload = {
            "prompt": actual_prompt_text,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        proxies = {}
        if self.http_proxy:
            proxies['http'] = self.http_proxy
        if self.https_proxy:
            proxies['https'] = self.https_proxy

        auth = HTTPBasicAuth(self.username, self.password)

        try:
            response = requests.post(
                self.url,
                headers=headers,
                json=payload,
                auth=auth,
                proxies=proxies if proxies else None,
                verify=False  # CRITICAL: As per requirements
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request failed: {e}")

        if self.evaluation_mode:
            # In evaluation mode, assume the response text is a JSON string
            # matching JudgeResponse structure. Return it directly.
            return response.text
        else:
            # Normal generation mode: parse JSON and extract text
            try:
                response_json = response.json()
                # Attempt common response structures
                if 'choices' in response_json and isinstance(response_json['choices'], list) and len(response_json['choices']) > 0:
                    if 'text' in response_json['choices'][0]:
                        return response_json['choices'][0]['text']
                    elif 'message' in response_json['choices'][0] and 'content' in response_json['choices'][0]['message']: # Anthropic style
                        return response_json['choices'][0]['message']['content']
                elif 'text' in response_json: # Simpler direct text response
                    return response_json['text']
                elif 'generated_text' in response_json: # Another common variant
                    return response_json['generated_text']
                else:
                    # Add more specific error about what was expected vs received if possible
                    raise ValueError(f"Unexpected response JSON structure during generation: {response_json}")
            except ValueError as e: # Catches JSON decoding errors from response.json() or ValueErrors raised above
                raise ValueError(f"Failed to parse response JSON or find text during generation: {e}. Response content: {response.text}")
            except KeyError as e: # Catches KeyErrors if expected keys are missing
                 raise ValueError(f"Missing expected key in response JSON during generation: {e}. Response content: {response.text}")


    def __str__(self) -> str:
        return f"CustomRequestsModel(url='{self.url}', model_id='{self.model_id}', evaluation_mode={self.evaluation_mode})"

    def __repr__(self) -> str:
        return self.__str__()
