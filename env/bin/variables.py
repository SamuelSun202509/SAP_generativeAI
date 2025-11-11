# TODO Change the resource group, embedding model ID and LLM/Chat model ID to your own
RESOURCE_GROUP = "default" # your resource group e.g. team-01
# RESOURCE_GROUP = "DocumentGrounding" # your resource group e.g. team-01
EMBEDDING_DEPLOYMENT_ID = "d12a770eac7c91c9" # e.g. d6b74feab22bc49a
LLM_DEPLOYMENT_ID = ""

# TODO Change the URL to your own orchestration deployment ID
# AICORE_ORCHESTRATION_DEPLOYMENT_URL = "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d02c3ee65a3a9f5c"
AICORE_ORCHESTRATION_DEPLOYMENT_URL = "https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/dcdc91cb8f73b02d"
# HANA table name to store your embeddings
# TODO Change the name to your team name
EMBEDDING_TABLE = "EMBEDDINGS_CODEJAM_"+"->ADD_YOUR_NAME_HERE<-"
