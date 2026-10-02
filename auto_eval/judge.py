"""Judge model for auto eval.

Authors:
    Muhammad Hilal Darul Fauzan (hilalfauzan9@gmail.com)

References:
    [1] https://gdplabs.gitbook.io/sdk/gen-ai-sdk/tutorials/inference/lm-invoker
"""

import os

from gllm_evals.utils.retry_configuration import get_default_retry_config
from gllm_inference.lm_invoker import build_lm_invoker
from gllm_inference.lm_invoker.lm_invoker import BaseLMInvoker

DEFAULT_JUDGE = "openrouter/google/gemini-3.1-flash-lite"
JUDGE_HYPERPARAMETERS = {"temperature": 0.0, "top_p": 0.001}
PRICE_PER_MILLION_TOKENS = {
    "openrouter/google/gemini-3.1-flash-lite": (0.25, 1.50),
    "openrouter/google/gemini-3-flash-preview": (0.50, 3.00),
}
_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def build_judge(model_id: str = DEFAULT_JUDGE) -> BaseLMInvoker:
    """Build the judge at temperature 0 and top_p 0.001, the evaluation settings of the platform benchmark.

    Args:
        model_id (str, optional): The judge model id, written as `<provider>/<model>`. Defaults to `DEFAULT_JUDGE`.

    Returns:
        BaseLMInvoker: The judge.

    Raises:
        RuntimeError: If the API key of the judge's provider is not set.
    """
    key_name = _PROVIDER_KEYS.get(model_id.split("/", 1)[0])
    credentials = os.getenv(key_name) if key_name else None
    if key_name and not credentials:
        raise RuntimeError(f"Set {key_name} in .env or in the environment to use {model_id}.")
    return build_lm_invoker(
        model_id=model_id,
        credentials=credentials,
        config={"retry_config": get_default_retry_config(), "default_hyperparameters": JUDGE_HYPERPARAMETERS},
    )
