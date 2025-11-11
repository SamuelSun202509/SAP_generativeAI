import logging
from typing import Dict, Optional, List, Any

from botocore.config import Config
from langchain_aws import ChatBedrock as ChatBedrock_, ChatBedrockConverse as ChatBedrockConverse_
from langchain_community.embeddings import BedrockEmbeddings as BedrockEmbeddings_
from pydantic import BaseModel, ConfigDict, model_validator

from gen_ai_hub.proxy.core.base import BaseProxyClient
from gen_ai_hub.proxy.gen_ai_hub_proxy.client import Deployment
from gen_ai_hub.proxy.langchain.init_models import catalog
from gen_ai_hub.proxy.native.amazon.clients import Session

MODEL_NAME_TO_MODEL_ID_MAP = {
    "amazon--titan-embed-text": "amazon.titan-embed-text-v1",
    "anthropic--claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "anthropic--claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
    "anthropic--claude-4-opus": "anthropic.claude-4-opus-20250514-v1:0",
    "anthropic--claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic--claude-3.7-sonnet": "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic--claude-4-sonnet":"anthropic.claude-sonnet-4-20250514-v1:0",
    "amazon--nova-micro": "amazon.nova-micro-v1:0",
    "amazon--nova-lite": "amazon.nova-lite-v1:0",
    "amazon--nova-pro": "amazon.nova-pro-v1:0",
    "amazon--nova-premier": "amazon.nova-premier-v1:0",
}


class AICoreBedrockBaseModel(BaseModel):
    """AICoreBedrockBaseModel provides all adjustments
    to boto3 based LangChain classes to enable communication
    with SAP AI Core."""

    model_config = ConfigDict(extra='allow')

    def __init__(
            self,
            *args,
            model_id: str = "",
            deployment_id: str = "",
            model_name: str = "",
            config_id: str = "",
            config_name: str = "",
            proxy_client: Optional[BaseProxyClient] = None,
            **kwargs,
    ):
        """Extends the constructor of the base class with aicore specific parameters."""
        client_params = {
            "deployment_id": deployment_id,
            "model_name": model_name,
            "config_id": config_id,
            "config_name": config_name,
            "proxy_client": proxy_client,
        }
        kwargs["client_params"] = client_params
        super().__init__(*args, model_id=model_id, **kwargs)

    @classmethod
    def get_corresponding_model_id(cls, model_name):
        if model_name not in MODEL_NAME_TO_MODEL_ID_MAP:
            raise ValueError("Model specified is not supported.")
        return MODEL_NAME_TO_MODEL_ID_MAP[model_name]

    # pylint: disable=no-self-argument
    @model_validator(mode='before')
    def validate_environment(cls, values: Dict) -> Dict:
        client_params = values.get("client_params")
        if not client_params and "model_kwargs" in values and isinstance(values["model_kwargs"], dict):
            client_params = values["model_kwargs"].get("client_params")
        
        if client_params and not values.get("client"):
            if "config" in values and values["config"] is not None:
                client_params["config"] = values["config"]
            values["client"] = Session().client(**client_params)
        
        if values.get('model_id') in (None, ''):
            values["model_id"] = AICoreBedrockBaseModel.get_corresponding_model_id(
                values["client"].aicore_deployment.model_name
            )
        
        # Remove client_params from model_kwargs to prevent it from being passed to AWS API
        if "model_kwargs" in values and isinstance(values["model_kwargs"], dict):
            values["model_kwargs"].pop("client_params", None)
        
        # Remove client_params from top level to prevent it from being passed to AWS API
        values.pop("client_params", None)

        return values


class ChatBedrock(AICoreBedrockBaseModel, ChatBedrock_):
    """Drop-in replacement for LangChain ChatBedrock."""

    model_config = ConfigDict(extra='allow')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ChatBedrockConverse(AICoreBedrockBaseModel, ChatBedrockConverse_):
    """Drop-in replacement for LangChain ChatBedrockConverse."""

    model_config = ConfigDict(extra='allow')

    def __init__(self, *args, **kwargs):
        self.extract_model_kwargs_parameters(kwargs)

        super().__init__(*args, **kwargs)

    def extract_model_kwargs_parameters(self, kwargs):
        # Extract parameters from model_kwargs to avoid circular reference issues
        model_kwargs = kwargs.get('model_kwargs', {})
        if isinstance(model_kwargs, dict):
            # Extract common parameters that should be passed directly
            for param_name in ['temperature', 'max_tokens', 'top_p', 'stop_sequences']:
                if param_name in model_kwargs and param_name not in kwargs:
                    kwargs[param_name] = model_kwargs.pop(param_name)

            # Clean up model_kwargs if it's now empty
            if not model_kwargs:
                kwargs.pop('model_kwargs', None)
            else:
                kwargs['model_kwargs'] = model_kwargs


class BedrockEmbeddings(AICoreBedrockBaseModel, BedrockEmbeddings_):
    """Drop-in replacement for LangChain BedrockEmbeddings."""

    model_config = ConfigDict(extra='allow')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def _build_bedrock_model_kwargs(
        deployment: Deployment,
        temperature: float,
        max_tokens: int,
        top_k: Optional[int],
        top_p: float,
        stop_sequences: Optional[List[str]],
        config: Optional[Config]
) -> Dict[str, Any]:
    """Builds the model_kwargs dictionary for Bedrock models."""
    if top_k:
        logging.warning(
            "Top-k is disabled for Amazon Bedrock models. Ignoring top-k value."
        )

    model_kwargs = {
        "temperature": temperature,
    }

    if config:
        model_kwargs["config"] = config

    if deployment.model_name.startswith("anthropic"):
        model_kwargs["max_tokens"] = max_tokens
        model_kwargs["top_p"] = top_p
    else:  # Assuming Amazon bedrock models otherwise
        model_kwargs["maxTokenCount"] = max_tokens
        model_kwargs["topP"] = top_p
        if stop_sequences:
            model_kwargs["stopSequences"] = stop_sequences

    return model_kwargs

@catalog.register(
    "gen-ai-hub",
    ChatBedrock,
    "anthropic--claude-3-haiku",
    "anthropic--claude-3-opus",
    "anthropic--claude-4-opus",
    "anthropic--claude-3.5-sonnet",
    "anthropic--claude-3.7-sonnet",
    "anthropic--claude-4-sonnet",
    "amazon--nova-premier",
)
def init_chat_model(
        proxy_client: BaseProxyClient,
        deployment: Deployment,
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        stop_sequences: List[str] = None,
        model_id: Optional[str] = '',
        config: Optional[Config] = None
):
    """Initializes a chat model using the legacy Bedrock Invoke API (`ChatBedrock`)."""
    model_kwargs = _build_bedrock_model_kwargs(
        deployment=deployment,
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=top_k,
        top_p=top_p,
        stop_sequences=stop_sequences,
        config=config
    )

    return ChatBedrock(
        model_name=deployment.model_name,
        model_id=model_id,
        deployment_id=deployment.deployment_id,
        proxy_client=proxy_client,
        model_kwargs=model_kwargs
    )

@catalog.register(
    "gen-ai-hub",
    ChatBedrockConverse,
    "anthropic--claude-3.7-sonnet",
    "anthropic--claude-4-sonnet",
)
def init_chat_converse_model(
        proxy_client: BaseProxyClient,
        deployment: Deployment,
        temperature: float = 0.0,
        max_tokens: int = 256,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        stop_sequences: List[str] = None,
        model_id: Optional[str] = '',
        config: Optional[Config] = None
):
    """
    Initializes a chat model using the newer Bedrock Converse API (`ChatBedrockConverse`).

    The Converse API offers several advantages over the older Invoke API:
    - Unified interface for different models and modalities.
    - Native support for tool use (function calling).
    - Standardized request/response structure.
    """

    return ChatBedrockConverse(
        model_name=deployment.model_name,
        model_id=model_id,
        deployment_id=deployment.deployment_id,
        proxy_client=proxy_client,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop_sequences=stop_sequences,
        config = config
    )

@catalog.register(
    "gen-ai-hub",
    BedrockEmbeddings,
    "amazon--titan-embed-text"
)
def init_embedding_model(proxy_client: BaseProxyClient, deployment: Deployment, model_id: Optional[str] = ''):
    return BedrockEmbeddings(deployment_id=deployment.deployment_id, proxy_client=proxy_client, model_id=model_id)
