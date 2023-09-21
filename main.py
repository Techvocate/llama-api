import os
import sys
import logging

from llama_index import (
    SimpleDirectoryReader, 
    StorageContext,
    load_index_from_storage, 
    VectorStoreIndex, 
    ServiceContext,
)
from llama_index.agent import OpenAIAgent
from llama_index.retrievers import VectorIndexRetriever
from llama_index.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores import SimpleVectorStore
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer

from langchain.chat_models import ChatOpenAI


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

# print(1) #<----------- For Debugging

storage_context = StorageContext.from_defaults(persist_dir="tpa/")

# print(2) #<----------- For Debugging

index = load_index_from_storage(
    storage_context=storage_context, index_id="vector_index")

# print(3) #<----------- For Debugging

template = """
You are a legal professional with experience of more than 30 years. 
You must draft a legal document as asked in the input context using plain language and 
easy to understand terms but ensure accuracy and completeness of the legal document. 
The Agent will tell you the relevant inforamtion like parties involved, terms of agreement 
and other necessary details.
The tools provided to you have smart interpretibility if you specify keywords 
in your query to the tool [Example a query for preparing agreements related to sale of a 
property, business, lease or rent should mention property agreemet, business agreements, 
lease agreement or rent agreement respectively].Think from the point of view of Indian Legal System.
"""

vector_retriever = VectorIndexRetriever(
    index=index, similarity_top_k=2, vector_store_query_mode=VectorStoreQueryMode.DEFAULT)

response_synthesizer = get_response_synthesizer(
    service_context=service_context,
    # response_mode="compact",
)
query_engine = RetrieverQueryEngine(
    vector_retriever,
    # response_synthesizer=response_synthesizer,
)

print(query_engine.query("Write an agreement to sale of a commercial property of size 20x40 at a price of Rs.50,00,000.00 and has no pending lawsuit."))


"""
Promp 1:
Write an agreement to sale of a commercial property of size 20x40 at a price of Rs.50,00,000.00 and has no pending lawsuit.

Prompt 2:
Draft a lease agreement for a residential property for minimum 8 months with the advance rent of 2 months and the decided rent is Rs.10,000 per month.
"""
