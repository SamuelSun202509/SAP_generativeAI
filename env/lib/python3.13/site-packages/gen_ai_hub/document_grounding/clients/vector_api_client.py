import humps
import requests
from typing import Optional

from gen_ai_hub import GenAIHubProxyClient
from gen_ai_hub.proxy import get_proxy_client
from pydantic import TypeAdapter

from ..models.vector import (
    CollectionCreateRequest,
    Collection,
    CollectionsListResponse,
    DocumentsCreateRequest,
    DocumentsUpdateRequest,
    Document,
    DocumentsResponse,
    DocumentsListResponse, CollectionCreationStatusResponse, CollectionDeletionStatusResponse, TextSearchRequest,
    VectorSearchResults,
)

# Constants
PATH_DOCUMENT_GROUNDING_VECTOR = "/lm/document-grounding/vector"


class VectorAPIClient:
    """
    The Vector API provides management and search capabilities for vector-based document collections.

    It enables creating, retrieving, updating, and deleting collections, as well as
    managing documents and performing semantic vector searches within those collections.

    Reference: https://api.sap.com/api/DOCUMENT_GROUNDING_API/resource/Vector

    Args:
        proxy_client: The proxy client to use for making requests.
    """

    def __init__(self, proxy_client: Optional[GenAIHubProxyClient] = None):
        """
        Initializes the VectorAPIClient
        Args:
            proxy_client: optional proxy client to use for requests
        """
        self.proxy_client = proxy_client or get_proxy_client(proxy_version="gen-ai-hub")
        self.rest_client = self.proxy_client.ai_core_client.rest_client
        self.path = PATH_DOCUMENT_GROUNDING_VECTOR

    # --- Collections ---

    def get_collections(
            self, top: Optional[int] = None, skip: Optional[int] = None, count: Optional[bool] = None
    ) -> CollectionsListResponse:
        """Get all collections."""
        params = {}
        if top is not None:
            params["$top"] = top
        if skip is not None:
            params["$skip"] = skip
        if count is not None:
            params["$count"] = count
        response = self.rest_client.get(path=f"{self.path}/collections", params=params)
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        return CollectionsListResponse(**response)

    def create_collection(self, collection_request: CollectionCreateRequest) -> requests.Response:
        """Create new collection
        Args:
            collection_request: The object containing the collection configuration.
        Returns:
             requests.Response empty object with 202 status code
        """
        response = self.rest_client.post(
            path=f"{self.path}/collections",
            body=collection_request.model_dump(exclude_none=True)
        )
        if response == "":  # rest_client (ai api sdk) returns empty string for 202 No Content
            response = requests.Response()
            response.status_code = 202
        return response

    def get_collection_by_id(self, collection_id: str) -> Collection:
        """Get collection details by ID."""
        response = self.rest_client.get(path=f"{self.path}/collections/{collection_id}")
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        return Collection(**response)

    def delete_collection(self, collection_id: str) -> requests.Response:
        """Delete collection by ID (async, check status)."""
        response = self.rest_client.delete(path=f"{self.path}/collections/{collection_id}")
        if response == "":  # rest_client (ai api sdk) returns empty string for 204 No Content
            response = requests.Response()
            response.status_code = 204
        return response

    # --- Documents ---

    def get_documents(self, collection_id: str, top: Optional[int] = None,
                      skip: Optional[int] = None, count: Optional[bool] = None) -> DocumentsResponse:
        """Get documents from a collection."""
        params = {}
        if top is not None:
            params['$top'] = top
        if skip is not None:
            params['$skip'] = skip
        if count is not None:
            params['$count'] = count
        response = self.rest_client.get(
            path=f"{self.path}/collections/{collection_id}/documents",
            params=params
        )
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        return DocumentsResponse(**response)

    def create_documents(self, collection_id: str, request: DocumentsCreateRequest) -> DocumentsListResponse:
        """Create new documents in a collection."""
        response = self.rest_client.post(
            path=f"{self.path}/collections/{collection_id}/documents",
            body=request.model_dump(exclude_none=True)
        )
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        return DocumentsListResponse(**response)

    def update_documents(self, collection_id: str, request: DocumentsUpdateRequest) -> DocumentsListResponse:
        """Update (upsert) documents in a collection."""
        response = self.rest_client.patch(
            path=f"{self.path}/collections/{collection_id}/documents",
            body=request.model_dump(exclude_none=True)
        )
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        return DocumentsListResponse(**response)

    def get_document_by_id(self, collection_id: str, document_id: str) -> Document:
        """Get a document by ID from a collection."""
        response = self.rest_client.get(
            path=f"{self.path}/collections/{collection_id}/documents/{document_id}"
        )
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        return Document(**response)

    def delete_document(self, collection_id: str, document_id: str) -> requests.Response:
        """Delete a document from a collection."""
        response = self.rest_client.delete(
            path=f"{self.path}/collections/{collection_id}/documents/{document_id}"
        )
        if response == "":  # 204 No Content
            response = requests.Response()
            response.status_code = 204
        return response

    # --- Collection statuses ---

    def get_collection_creation_status(self, collection_id: str) -> CollectionCreationStatusResponse:
        """Get creation status for a collection."""
        response = self.rest_client.get(path=f"{self.path}/collections/{collection_id}/creationStatus")
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        adapter = TypeAdapter(CollectionCreationStatusResponse)
        return adapter.validate_python(response)

    def get_collection_deletion_status(self, collection_id: str) -> CollectionDeletionStatusResponse:
        """Get deletion status for a collection."""
        response = self.rest_client.get(path=f"{self.path}/collections/{collection_id}/deletionStatus")
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        adapter = TypeAdapter(CollectionDeletionStatusResponse)
        return adapter.validate_python(response)

    # --- Search ---

    def search(self, request: TextSearchRequest) -> VectorSearchResults:
        """Perform semantic search in vector collections."""
        response = self.rest_client.post(
            path=f"{self.path}/search",
            body=request.model_dump(exclude_none=True)
        )
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        return VectorSearchResults(**response)
