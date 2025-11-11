import humps
from typing import Optional

from gen_ai_hub import GenAIHubProxyClient
from gen_ai_hub.proxy import get_proxy_client

from ..models.retrieval import (
    DataRepositories,
    DataRepository,
    RetrievalSearchInput,
    RetrievalSearchResults,
)

# Constants
PATH_DOCUMENT_GROUNDING_RETRIEVAL = "/lm/document-grounding/retrieval"


class RetrievalAPIClient:
    """
    The Retrieval API enables querying and retrieving relevant content from configured data repositories,
    such as vector or external document sources (e.g., help.sap.com).

    Retrieval combines semantic search with repository metadata filtering and supports custom
    retrieval configurations for chunk/document granularity.

    Reference: https://api.sap.com/api/DOCUMENT_GROUNDING_API/resource/Retrieval

    Args:
        proxy_client: The proxy client to use for making requests.
    """

    def __init__(
        self,
        proxy_client: Optional[GenAIHubProxyClient] = None,
    ):
        """
        Initialize the RetrievalAPIClient.
        Args:
            proxy_client: Optional proxy client for making API requests.
        """
        self.proxy_client = proxy_client or get_proxy_client(proxy_version="gen-ai-hub")
        self.rest_client = self.proxy_client.ai_core_client.rest_client
        self.path = PATH_DOCUMENT_GROUNDING_RETRIEVAL

    # --- Data Repositories ---

    def get_data_repositories(
        self,
        top: Optional[int] = None,
        skip: Optional[int] = None,
        count: Optional[bool] = None,
    ) -> DataRepositories:
        """
        List all data repositories available to the tenant.
        """
        params = {}
        if top is not None:
            params["$top"] = top
        if skip is not None:
            params["$skip"] = skip
        if count is not None:
            params["$count"] = count

        response = self.rest_client.get(path=f"{self.path}/dataRepositories", params=params)
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        return DataRepositories(**response)

    def get_data_repository_by_id(self, repository_id: str) -> DataRepository:
        """
        Get a single data repository by its unique ID.
        """
        response = self.rest_client.get(path=f"{self.path}/dataRepositories/{repository_id}")
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        return DataRepository(**response)

    # --- Search ---

    def search(self, search_input: RetrievalSearchInput) -> RetrievalSearchResults:
        """
        Perform a retrieval search for relevant content given a query string and filters.

        Args:
            search_input: RetrievalSearchInput model defining the query and filters.

        Returns:
            RetrievalSearchResults model containing repositories, documents, and chunks.
        """
        response = self.rest_client.post(
            path=f"{self.path}/search",
            body=search_input.model_dump(exclude_none=True),
        )

        response = humps.camelize(response) # rest_client (ai api sdk) returns snake_case responses
        return RetrievalSearchResults(**response)