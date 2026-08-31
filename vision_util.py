# File: vision_util.py

import logging
import time
from pathlib import Path


class VisionUtil:
    """Describes images using either a local Ollama model or a remote
    OpenAI-compatible vision API (Venice, OpenAI, ...).

    The backend is selected by the ``provider`` key of ``vision_config``
    (default ``ollama``, which keeps existing configs working unchanged).
    """

    def __init__(self, vision_config: dict):
        self.logger = logging.getLogger('vision_util')
        provider = vision_config.get('provider', 'ollama')

        if provider in ('venice', 'openai'):
            from openai_vision_client import OpenAIVisionClient

            default_base = (
                'https://api.venice.ai/api/v1'
                if provider == 'venice'
                else 'https://api.openai.com/v1'
            )
            self.client = OpenAIVisionClient(
                model=vision_config['model'],
                api_base=vision_config.get('apiBase', default_base),
                api_key=vision_config.get('apiKey'),
                timeout=vision_config.get('request_timeout', 180),
                max_tokens=vision_config.get('max_tokens', 400),
                temperature=vision_config.get('temperature', 0.2),
            )
        elif provider == 'ollama':
            from ollama_client import OllamaClient

            self.client = OllamaClient(
                model=vision_config['model'],
                url=vision_config.get('endpoint', 'http://localhost:11434'),
            )
        else:
            raise ValueError(f"Unsupported vision provider: {provider}")

        self.logger.debug(
            f"Vision backend: provider={provider}, model={vision_config['model']}"
        )

    def describe_image(self, image_path: str, prompt: str) -> str:
        """
        Describe an image using the configured vision backend, with retry logic.

        Args:
            image_path (str): The file path to the image.
            prompt (str): The prompt to guide the description.

        Returns:
            str: The description of the image or a default message if failed.
        """
        retries = 3
        delays = [2, 4, 8]  # Delays in seconds for each retry

        for attempt in range(1, retries + 1):
            try:
                self.logger.debug(
                    f"Attempt {attempt}: Describing image at path '{image_path}' with prompt:\n{prompt}"
                )
                response = self.client.generate_with_image(
                    prompt=prompt, image_path=image_path
                )
                description = response.strip()
                self.logger.debug(f"Received image description:\n{description}")
                return description

            except Exception as e:
                self.logger.error(
                    f"Vision API request failed on attempt {attempt} for image '{image_path}': {e}"
                )
                if attempt < retries:
                    delay = delays[attempt - 1]
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"All {retries} attempts failed for image '{image_path}'. Skipping this image."
                    )
                    # Return a default description or handle as needed
                    return "Description unavailable due to an error."
