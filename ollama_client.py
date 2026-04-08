# File: ollama_client.py

import logging
import time
import ollama
from pathlib import Path


class OllamaClient:
    def __init__(self, model, url="http://localhost:11434", timeout=600):
        self.model = model
        self.client = ollama.Client(host=url, timeout=timeout)
        self.logger = logging.getLogger("ollama_client")
        self.max_retries = 5
        self.initial_wait = 3

    def generate(self, prompt):
        wait_time = self.initial_wait
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Sending prompt to Ollama LLM:\n{prompt}")
                response = self.client.generate(model=self.model, prompt=prompt)
                self.logger.debug(
                    f"Received response from Ollama LLM:\n{response['response']}"
                )
                return response["response"]
            except Exception as e:
                self.logger.warning(
                    f"Ollama API request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    self.logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    self.logger.error(
                        f"Ollama API request failed after {self.max_retries} attempts: {e}"
                    )
                    raise Exception(
                        f"Ollama API request failed after {self.max_retries} attempts: {str(e)}"
                    )

    def generate_with_image(self, prompt, image_path):
        wait_time = self.initial_wait
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(
                    f"Sending prompt with image to Ollama LLM:\nPrompt:\n{prompt}\nImage Path: {image_path}"
                )
                response = self.client.generate(
                    model=self.model, prompt=prompt, images=[Path(image_path)]
                )
                self.logger.debug(
                    f"Received response from Ollama LLM:\n{response['response']}"
                )
                return response["response"]
            except Exception as e:
                self.logger.warning(
                    f"Ollama API request failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    self.logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    self.logger.error(
                        f"Ollama API request failed after {self.max_retries} attempts: {e}"
                    )
                    raise Exception(
                        f"Ollama API request failed after {self.max_retries} attempts: {str(e)}"
                    )
