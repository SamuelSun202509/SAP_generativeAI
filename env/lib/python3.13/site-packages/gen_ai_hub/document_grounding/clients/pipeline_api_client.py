import humps
import requests
from typing import Optional

from gen_ai_hub import GenAIHubProxyClient
from gen_ai_hub.proxy import get_proxy_client

from ..models.pipeline import (
    CreatePipelineRequest,
    PipelineIdResponse,
    GetPipelinesResponse,
    GetPipelineStatusResponse,
    BasePipelineResponse,
    SearchPipelineRequest,
    SearchPipelinesResponse,
    GetPipelineExecutionsResponse,
    PipelineExecution,
    Document,
    DocumentsStatusResponse,
    ManualPipelineTrigger,
)

# Constants
PATH_DOCUMENT_GROUNDING = "/lm/document-grounding/pipelines"
PATH_DOCUMENT_GROUNDING_PIPELINES = PATH_DOCUMENT_GROUNDING # PATH_DOCUMENT_GROUNDING is kept for backward compatibility

class PipelineAPIClient:
    """The Pipelines API creates and manages vector stores based on documents from user data repositories:
    S3, SFTP, and Microsoft SharePoint.
    Each pipeline represents a configured end-to-end process including the following steps:
    - Fetches documents from a supported data source
    - Preprocesses and chunks the document content, and generates semantic embeddings.
      Semantic embeddings are multidimensional representations of textual information.
    - Stores semantic embeddings into the HANA Vector Store

    The Pipeline API is compatible with the following data repositories:
    - Microsoft SharePoint
    - AWS S3
    - SFTP


    see https://api.sap.com/api/DOCUMENT_GROUNDING_API/resource/Pipelines

    Args:
        proxy_client: The proxy client to use for making requests.
    """

    def __init__(
            self,
            proxy_client: Optional[GenAIHubProxyClient] = None,
    ):
        """
        Initializes the PipelineAPIClient
        Args:
            proxy_client: optional proxy client to use for requests
        """
        self.proxy_client = proxy_client or get_proxy_client(proxy_version="gen-ai-hub")
        self.rest_client = self.proxy_client.ai_core_client.rest_client
        self.path = PATH_DOCUMENT_GROUNDING

    def create_pipeline(self, pipeline_request: CreatePipelineRequest) -> PipelineIdResponse:
        """Create a document vectorization pipeline
        Args:
            pipeline_request: The object containing the pipeline configuration.
        Returns:
            PipelineIdResponse object containing the ID of the created pipeline.
        """
        response = self.rest_client.post(path=self.path, body=pipeline_request.model_dump(exclude_none=True))
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        return PipelineIdResponse(**response)

    def get_pipelines(self, top: Optional[int] = None, skip: Optional[int] = None, count: Optional[bool] = None) \
            -> GetPipelinesResponse:
        """Get all pipelines."""
        params = {}
        if top is not None:
            params['$top'] = top
        if skip is not None:
            params['$skip'] = skip
        if count is not None:
            params['$count'] = count
        response = self.rest_client.get(path=self.path, params=params)
        return GetPipelinesResponse(**response)

    def get_pipeline_by_id(self, pipeline_id: str) -> BasePipelineResponse:
        """Get details of a pipeline by pipeline id."""
        response = self.rest_client.get(path=f"{self.path}/{pipeline_id}")
        return BasePipelineResponse(**response)

    def delete_pipeline_by_id(self, pipeline_id: str) -> requests.Response:
        """Delete a pipeline by pipeline id."""
        response = self.rest_client.delete(path=f"{self.path}/{pipeline_id}")
        if response == "":  # rest_client (ai api sdk) returns empty string for 204 No Content
            response = requests.Response()
            response.status_code = 204
        return response

    def get_pipeline_status(self, pipeline_id: str) -> GetPipelineStatusResponse:
        """Get pipeline status by pipeline id."""
        response = self.rest_client.get(path=f"{self.path}/{pipeline_id}/status")
        response = humps.camelize(response)  # rest_client (ai api sdk) returns snake_case responses
        return GetPipelineStatusResponse(**response)

    def search_pipelines(self, body: SearchPipelineRequest) -> SearchPipelinesResponse:
        """Pipeline Search by Metadata"""
        response = self.rest_client.post(path=f"{self.path}/search", body=body.model_dump(exclude_none=True))
        response = humps.camelize(response)
        return SearchPipelinesResponse(**response)

    # pylint: disable=unused-import
    def get_pipeline_executions(
        self,
        pipeline_id: str,
        last_execution: Optional[bool] = None,
        top: Optional[int] = None,
        skip: Optional[int] = None,
        count: Optional[bool] = None,
    ) -> GetPipelineExecutionsResponse:
        """Get Pipeline Executions"""
        params = {}
        if last_execution is not None:
            params["lastExecution"] = last_execution
        if top is not None:
            params["$top"] = top
        if skip is not None:
            params["$skip"] = skip
        if count is not None:
            params["$count"] = count
        response = self.rest_client.get(path=f"{self.path}/{pipeline_id}/executions", params=params)
        response = humps.camelize(response)
        return GetPipelineExecutionsResponse(**response)

    def get_pipeline_execution_by_id(self, pipeline_id: str, execution_id: str) -> PipelineExecution:
        """Get Pipeline Execution by ID"""
        response = self.rest_client.get(path=f"{self.path}/{pipeline_id}/executions/{execution_id}")
        response = humps.camelize(response)
        return PipelineExecution(**response)

    # pylint: disable=too-many-arguments
    def get_execution_documents(
        self,
        pipeline_id: str,
        execution_id: str,
        top: Optional[int] = None,
        skip: Optional[int] = None,
        count: Optional[bool] = None,
    ) -> DocumentsStatusResponse:
        """Get Documents for a Pipeline Execution"""
        params = {}
        if top is not None:
            params["$top"] = top
        if skip is not None:
            params["$skip"] = skip
        if count is not None:
            params["$count"] = count
        response = self.rest_client.get(
            path=f"{self.path}/{pipeline_id}/executions/{execution_id}/documents",
            params=params,
        )
        response = humps.camelize(response)
        return DocumentsStatusResponse(**response)

    def get_execution_document_by_id(self, pipeline_id: str, execution_id: str, document_id: str) -> \
            Document:
        """Get Document by ID for a Pipeline Execution"""
        response = self.rest_client.get(
            path=f"{self.path}/{pipeline_id}/executions/{execution_id}/documents/{document_id}"
        )
        response = humps.camelize(response)
        return Document(**response)

    def get_pipeline_documents(
        self,
        pipeline_id: str,
        top: Optional[int] = None,
        skip: Optional[int] = None,
        count: Optional[bool] = None,
    ) -> DocumentsStatusResponse:
        """Get Documents for a Pipeline"""
        params = {}
        if top is not None:
            params["$top"] = top
        if skip is not None:
            params["$skip"] = skip
        if count is not None:
            params["$count"] = count
        response = self.rest_client.get(path=f"{self.path}/{pipeline_id}/documents", params=params)
        response = humps.camelize(response)
        return DocumentsStatusResponse(**response)

    def get_pipeline_document_by_id(self, pipeline_id: str, document_id: str) -> Document:
        """Get Document by ID for a Pipeline"""
        response = self.rest_client.get(path=f"{self.path}/{pipeline_id}/documents/{document_id}")
        response = humps.camelize(response)
        return Document(**response)

    def trigger_pipeline(self, request: ManualPipelineTrigger) -> requests.Response:
        """Pipeline Trigger"""
        response = self.rest_client.post(path=f"{self.path}/trigger", body=request.model_dump(exclude_none=True))
        if response == "":  # rest_client (ai api sdk) returns empty string for 204 No Content
            response = requests.Response()
            response.status_code = 202
        return response
