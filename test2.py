import os
import logging
import sys
import tiktoken

from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index import StorageContext, load_index_from_storage, SummaryIndex, ServiceContext
from llama_index.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.vector_stores.types import VectorStoreQueryMode
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.schema import IndexNode
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI

# Llama-Lang
from llama_index.langchain_helpers.agents.tools import IndexToolConfig, LlamaIndexTool

from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser, load_tools, initialize_agent, AgentType
# from langchain import LLMChain, OpenAI


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)

callback_manager = CallbackManager([token_counter])

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

#!-------------------------------------
index_name = "embeddings"
environment = "us-west4-gcp-free"
pinecone.init(api_key=PINECONE_API_KEY, environment=environment)

pinevs = PineconeVectorStore(
    index_name=index_name,
    environment=environment,
    service_context = service_context,
    # metadata_filters=metadata_filters,
)

pineStorageContext = StorageContext.from_defaults(vector_store=pinevs)

pineIndex = VectorStoreIndex.from_vector_store(vector_store= pinevs)
# pineRetriever = VectorIndexRetriever(
#     index=pineIndex, similarity_top_k=2, vector_store_query_mode=VectorStoreQueryMode.DEFAULT
# )

pineResponse_synthesizer = get_response_synthesizer(
    service_context=service_context,
    # response_mode="compact",
)

pine_query_engine = pineIndex.as_query_engine(template = PromptTemplate(template1), )
# pine_query_engine = RetrieverQueryEngine.from_args(
#     pineRetriever,
#     # response_synthesizer=response_synthesizer,
#     service_context=service_context,
# )

# response = query_engine.query("What is the filename for Property transfer?")
# # define tools
# query_engine_tools = [QueryEngineTool(query_engine=pine_query_engine, metadata=ToolMetadata(
#     name="retriever_tool", description="Useful for summarization of doc"),),]


#!-------------------------------------
names = ["The Indian Contract Act, 1872","The Sales of Goods Act,1930","The Specific Relief Act, 1963","The Transfer of Property Act, 1882", "The Uttar Pradesh Urban Buildings (Regulation of Letting, Rent and Eviction) Act, 1972"]
descriptions = ["The go-to document for Contract Rules. The Indian Contract Act, 1872 is a fundamental legal framework in India that governs the formation and enforcement of contracts, defining the rules and principles that underlie agreements between parties in various transactions.", "The go-to document for rules related to sale of goods. The Sales of Goods Act, 1930 is an Indian legislation that governs the sale and purchase of goods, outlining the rights and obligations of both buyers and sellers in commercial transactions.", "The go-to document for Relief Rules. The Specific Relief Act, 1963 is an Indian legal statute that governs the remedies available for the enforcement of civil rights and obligations, emphasizing the specific performance of contracts as a primary remedy.", "The go-to document for Rules for transferring property and its rights. The Transfer of Property Act, 1882 is a legal statute in India that governs the transfer of property from one person to another. It defines various types of property transactions, including sales, mortgages, leases, and gifts, and sets out the legal rules and procedures for such transfers.", "The go-to document for Rules and Regulations for Rent in Uttar Pradesh. The Uttar Pradesh Urban Buildings Act of 1972 regulates the rental and eviction of urban properties in the Indian state of Uttar Pradesh"]
temp = ['ica', 'sga', 'sra', 'tpa', 'upra']

# query_engine_tools = []
agents = {}

for n,x in enumerate(temp):
    storage_context = StorageContext.from_defaults(persist_dir=x)
    
    vector_index = load_index_from_storage(storage_context=storage_context, index_id="vector_index")
    summary_index = load_index_from_storage(storage_context=storage_context, index_id="vector_index")
    
    vector_query_engine = vector_index.as_query_engine()
    list_query_engine = summary_index.as_query_engine()
    
    query_engine_tools = [
        QueryEngineTool(
            query_engine = vector_query_engine,
            metadata = ToolMetadata(name = temp[n],description= descriptions[n])
        ),
        QueryEngineTool(
            query_engine = list_query_engine,
            metadata =  ToolMetadata(name = temp[n], description= descriptions[n])
        )
    ]

    # tool_config = IndexToolConfig(
    #         query_engine=vector_query_engine, 
    #         name=f"Vector Index",
    #         description="useful for when you want to answer queries about {x}",
    #         tool_kwargs={"return_direct": True},
    #     ),
        # IndexToolConfig(
        #     query_engine = list_query_engine,
        #     name = f"Summary Index",
        #     descriptions = "Useful for retrieving specific context from {x}",
        #     tool_kwargs={"return_direct": True},
        # ),
    

    # tool_names = [tool.name for tool in tools_config]

    function_llm = OpenAI(model="gpt-3.5-turbo-0613")
    # tools = LlamaIndexTool.from_tool_config(tool_config)
    
    # tools = load_tools(query_engine_tools, llm=llm)
    # agent = initialize_agent(
    #     tools,
    #     llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    #     return_intermediate_steps=True,
    # )
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
    )
    agents[x] = agent

nodes = []
for n,x in enumerate(names):
    act_summary = (
        f"This content contains Acts about {x}. "
        f"Use this index if you need to lookup specific facts about {x}.\n"
        "Do not use this index if you want to analyze multiple acts."
    )
    node = IndexNode(text=act_summary, index_id=temp[n])
    nodes.append(node)

print(nodes)

vector_index = VectorStoreIndex(nodes)
vector_retriever = vector_index.as_retriever(similarity_top_k=1)

recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=agents,
    verbose=True,
)

response_synthesizer = get_response_synthesizer(
    # service_context=service_context,
    response_mode="compact",
)
query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer,
    service_context=service_context,
)
token_counter.reset_counts()
print(query_engine.query("Draft an agreement for sale of goods between two parties i.e M/s Elite Electronics and Bright Bulb Pvt. Ltd. Bright Bulb Pvt. Ltd. will supply material to M/s Elite Electronics as per their demand, but M/s Elite Electronics must have to buy minimum 500 units per month, not less than that and cost for each unit will be dependent on the material. The term period for this agreement will be of 15 months."))
print(token_counter.total_llm_token_count)

# print(query_engine.query("Write an agreement to sale of a commercial property of size 20x40 at a price of Rs.50,00,000.00 and has no pending lawsuit."))
# print(query_engine.query("Draft a lease agreement for a residential property for minimum 8 months with the advance rent of 2 months and the decided rent is Rs.10,000 per month."))