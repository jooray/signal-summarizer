# File: openai_vision_client.py

import base64
import logging
import mimetypes
import time
from pathlib import Path

import httpx


class OpenAIVisionClient:
    """Vision client for OpenAI-compatible chat APIs (Venice, OpenAI, ...).

    Mirrors the interface of OllamaClient.generate_with_image so VisionUtil can
    use either backend interchangeably.
    """

    def __init__(
        self,
        model,
        api_base="https://api.venice.ai/api/v1",
        api_key=None,
        timeout=180,
        max_tokens=400,
        temperature=0.2,
    ):
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger = logging.getLogger("openai_vision_client")
        self.max_retries = 5
        self.initial_wait = 3

    def _data_uri(self, image_path):
        path = Path(image_path)
        mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _extract_content(self, data):
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            # Some providers return content as a list of parts.
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return (content or "").strip()

    def generate_with_image(self, prompt, image_path):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": self._data_uri(image_path)},
                        },
                    ],
                }
            ],
            "max_completion_tokens": self.max_tokens,
            "temperature": self.temperature,
            "venice_parameters": {"include_venice_system_prompt": False},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        wait_time = self.initial_wait
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(
                    f"Sending prompt with image to {self.model}:\n"
                    f"Prompt:\n{prompt}\nImage Path: {image_path}"
                )
                response = httpx.post(
                    f"{self.api_base}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                content = self._extract_content(response.json())
                if not content:
                    raise ValueError("Vision API returned an empty description")
                self.logger.debug(f"Received response from vision model:\n{content}")
                return content
            except Exception as e:
                self.logger.warning(
                    f"Vision API request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    self.logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    self.logger.error(
                        f"Vision API request failed after {self.max_retries} attempts: {e}"
                    )
                    raise Exception(
                        f"Vision API request failed after {self.max_retries} attempts: {str(e)}"
                    )
