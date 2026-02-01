from langchain_openai import ChatOpenAI
from typing import Dict, Any


# This is a helper function that overrides some functionality from OpenAI API
# that is not imlemented by Venice.

class ChatVeniceAPI(ChatOpenAI):

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Venice does not understand the n parameter, so remove it."""
        params = super()._default_params  # Get the existing parameters
        if 'n' in params:
            del params['n']
        return params
