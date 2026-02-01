# File: llm_util.py

import logging
from typing import List, Optional, Type, Any

from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
import time


class LinkSummary(BaseModel):
    title: Optional[str] = Field(description="The title of the webpage")
    summary: Optional[str] = Field(description="A concise summary of the webpage")
    error: Optional[str] = Field(description="Error message if any occurred during summarization")


class ConversationTheme(BaseModel):
    name: str = Field(description="Name of the theme")
    summary: str = Field(description="Summary of the discussion on this theme")
    dissenting_opinions: Optional[str] = Field(description="Optional dissenting opinions on the topic")


class ConversationThemes(BaseModel):
    themes: List[ConversationTheme] = Field(description="List of themes discussed")


class SimilarityGroup(BaseModel):
    theme_numbers: List[int] = Field(description="List of theme numbers that are similar")
    similarity_rating: float = Field(description="Similarity rating for the group")


class SimilarityGroups(BaseModel):
    groups: List[SimilarityGroup] = Field(description="List of similarity groups")


class LLMUtil:
    def __init__(self, model_config):
        self.model_config = model_config
        self.provider = model_config.get('provider', 'ollama')
        self.logger = logging.getLogger('llm_util')

        if self.provider == 'ollama':
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(
                base_url=model_config.get('endpoint', 'http://localhost:11434'),
                model=model_config['model']
            )
        elif self.provider == 'openai':
            from langchain_openai import ChatOpenAI
            self.llm = ChatOpenAI(
                model=model_config['model'],
                api_key=model_config.get('apiKey'),
                base_url=model_config.get('apiBase')
            )
        elif self.provider == 'venice':
            from chat_venice_api import ChatVeniceAPI
            self.llm = ChatVeniceAPI(
                model=model_config['model'],
                api_key=model_config.get('apiKey'),
                base_url=model_config.get('apiBase')
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        max_retries = 5
        wait_time = 3

        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Sending prompt to LLM:\n{prompt}")
                response = self.llm.invoke(prompt)
                response = getattr(response, 'content', str(response))
                self.logger.debug(f"Received response from LLM:\n{response}")
                return response
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    self.logger.error(f"LLM API request failed after {max_retries} attempts: {e}")
                    raise Exception(f"LLM API request failed after {max_retries} attempts: {str(e)}")

    def split_text(self, text: str, max_chunk_size: int) -> List[str]:
        """Split text into manageable chunks for processing."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_text(text)
        self.logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks

    def generate_structured_output(
        self,
        prompt_template: str,
        context: str,
        pydantic_class: Type[BaseModel],
        max_retries: int = 3,
        **kwargs,
    ) -> BaseModel:
        """Generate structured output using a Pydantic model with OutputFixingParser."""
        # Initialize the Pydantic parser
        pydantic_parser = PydanticOutputParser(pydantic_object=pydantic_class)

        # Prepare format instructions
        format_instructions = pydantic_parser.get_format_instructions()

        # Prepare prompt data
        prompt_data = {
            "format_instructions": format_instructions,
            "context": context,
            **kwargs,
        }

        # Construct the full prompt
        prompt_template_obj = PromptTemplate(
            template=prompt_template,
            input_variables=list(prompt_data.keys()),
        )
        full_prompt = prompt_template_obj.format(**prompt_data)

        self.logger.debug(f"Full prompt sent to LLM:\n{full_prompt}")

        # Initialize the OutputFixingParser with the Pydantic parser and the same LLM
        output_fixer = OutputFixingParser.from_llm(
            parser=pydantic_parser,
            llm=self.llm,
            max_retries=max_retries,
        )

        try:
            # Use OutputFixingParser to parse and fix the output if necessary
            result = output_fixer.parse(full_prompt)
            self.logger.debug(f"Received structured output:\n{result}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to generate valid output: {e}")
            raise

    def summarize_link(
        self, context: str, prompt_template: str, max_title_length: int, max_summary_length: int
    ) -> LinkSummary:
        """Summarize a webpage or other content."""
        result = self.generate_structured_output(
            prompt_template=prompt_template,
            context=context,
            pydantic_class=LinkSummary,
            max_title_length=max_title_length,
            max_summary_length=max_summary_length,
            max_retries=3,
        )

        if result.error:
            return result

        title = result.title
        summary = result.summary
        if (
            title
            and len(title) <= max_title_length
            and "\n" not in title
            and summary
            and len(summary) <= max_summary_length
            and "\n" not in summary
        ):
            return result
        else:
            result.error = "Output did not meet length or format constraints."
            result.title = None
            result.summary = None
            return result

    def translate_text(
        self, text: str, target_language: str, prompt_template: str, max_chunk_size: int = 2000
    ) -> str:
        """Translate the given text into the target language using the provided language prompt."""
        if not prompt_template:
            raise ValueError("Language prompt is not provided.")

        chunks = self.split_text(text, max_chunk_size)
        translated_chunks = []

        for chunk in chunks:
            # Prepare the prompt
            prompt = prompt_template.format(language=target_language, context=chunk)
            try:
                translated_chunk = self.generate(prompt)
                self.logger.debug(f"Translated chunk:\n{translated_chunk}")
                translated_chunks.append(translated_chunk)
            except Exception as e:
                self.logger.error(f"Translation failed for a chunk: {e}")
                translated_chunks.append(chunk)  # Append the original chunk if translation fails

        # Recombine the translated chunks with double newlines to preserve markdown structure
        translated_text = '\n\n'.join(translated_chunks)
        return translated_text
