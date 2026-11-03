
# Cache allows us to reuse the same OpenAI client.
from functools import lru_cache

# Official OpenAI Python SDK.
from openai import OpenAI

# Application settings.
from app.core.config import settings


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAI:
    """
    Create and return a reusable OpenAI client.

    The API key is loaded from the application settings,
    which read sensitive values from the .env file.

    Returns:
        OpenAI: Configured OpenAI client.

    Raises:
        ValueError: If the API key is not configured.
    """

    # Check whether an API key is configured.
    if settings.openai_api_key is None:
        raise ValueError(
            "OPENAI_API_KEY is not configured. "
            "Please add it to your .env file."
        )

    # Extract the secret value only when initializing the client.
    api_key = settings.openai_api_key.get_secret_value()

    # Create the OpenAI client.
    return OpenAI(api_key=api_key)