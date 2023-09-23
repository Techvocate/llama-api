import logging
import os
import sys
import pinecone
from constants import PINECONE_API_KEY, pine_env, pine_index

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import PineconeVectorStore
# from IPython.display import Markdown, display

from llama_index.storage.storage_context import StorageContext

# api_key = os.environ[PINECONE_API_KEY]
# get pine_env , pine_index from the pinecone account
pinecone.init(api_key=PINECONE_API_KEY, environment=pine_env)


# pinecone.create_index("embeddings", dimension=1536, metric="cosine", pod_type="starter") 
pinecone_index = pinecone.Index(pine_index)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


documents = SimpleDirectoryReader("D:\\Downloads\\Acts\\Acts").load_data()

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, show_progress= True)

query_engine = index.as_query_engine()

