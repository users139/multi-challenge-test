import requests
from requests.auth import HTTPBasicAuth
from src.models.base import ModelProvider  # Assuming this path is correct

class CustomRequestsModel(ModelProvider):
    def __init__(self, url: str, model_id: str, username: str, password: str, temperature: float, max_tokens: int = None, top_p: float = 1.0, http_proxy: str = None, https_proxy: str = None):
        self.url = url
        self.model_id = model_id
        self.username = username
        self.password = password
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.http_proxy = http_proxy
        self.https_proxy = https_proxy

    def generate(self, prompt: str) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "prompt": prompt,
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
                raise ValueError(f"Unexpected response JSON structure: {response_json}")
        except (ValueError, KeyError) as e: # Catches JSON decoding errors and key errors
            raise ValueError(f"Failed to parse response JSON or find text: {e}. Response content: {response.text}")

    def __str__(self) -> str:
        return f"CustomRequestsModel(url='{self.url}', model_id='{self.model_id}')"

    def __repr__(self) -> str:
        return self.__str__()
