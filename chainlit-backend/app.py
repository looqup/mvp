import os
from openai import AsyncOpenAI
import getpass
from fastapi.responses import JSONResponse

from chainlit.auth import create_jwt
from chainlit.server import app
import chainlit as cl

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# from llama_index.core import SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
from IPython.display import Markdown, display
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core.tools import QueryEngineTool,ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

import pandas as pd 
from llama_index.core.query_engine import PandasQueryEngine
from llama_index.core.query_engine import RouterQueryEngine

os.environ[
    "PINECONE_API_KEY"
] = "34106af0-e289-4a1e-a502-57e430853030"
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
from pinecone import Pinecone
from pinecone import ServerlessSpec

api_key = os.environ["PINECONE_API_KEY"]
pc = Pinecone(api_key=api_key)
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


settings = {
    "model": "gpt-3.5-turbo",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}
# pc.create_index(
#     "htmlcsvindex1",
#     dimension=1536,
#     metric="euclidean",
#     spec=ServerlessSpec(cloud="aws", region="us-west-2"),
# )
pinecone_index = pc.Index("htmlcsvindex")
documents = SimpleWebPageReader(html_to_text=True).load_data(["https://www.fda.gov/safety/recalls-market-withdrawals-safety-alerts"])
documents1 = SimpleDirectoryReader("./data/").load_data()
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)
# index1 = VectorStoreIndex.from_documents(
#     documents1, storage_context=storage_context
# )

# csvIndex = index1.as_query_engine()
htmlIndex = index.as_query_engine()
df = pd.read_csv("./data/Customers.csv")
df2 = pd.read_csv("./data/Purchases.csv")
df3 = pd.read_csv("./data/Locations.csv")

# Pandas Merge on Multiple DataFrames using pd.merge()
df4 = pd.merge(pd.merge(df,df2,on='customer_id'),df3,on='location_id')

# By using DataFrame.merge()
df5 = df.merge(df2,on='customer_id').merge(df3,on='location_id')

# Merge multiple DataFrames using left join
df6 = df.merge(df2,how ='left').merge(df3,how ='left')
df6.head()
query_engine1 = PandasQueryEngine(df=df6,verbose=True)
tool1 =   QueryEngineTool(
    query_engine=query_engine1, 
    metadata=ToolMetadata(name='Purchases', description='List of customers with details, purchases made by them with their feedback and locations customer belogs to ')
)
tool2 =   QueryEngineTool(
    query_engine=htmlIndex, 
    metadata=ToolMetadata(name='recall', description='Provides information about products that have been called back '))

@app.get("/custom-auth")
async def custom_auth():
    # Verify the user's identity with custom logic.
    token = create_jwt(cl.User(identifier="Test User"))
    return JSONResponse({"token": token})

@cl.on_chat_start
async def on_chat_start():
    
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    query_engine2 = RouterQueryEngine.from_defaults(query_engine_tools=[tool1,tool2])
    cl.user_session.set("query_engine", query_engine2)

    await cl.Message(
        author="Assistant", content="Hello! Im an AI assistant. How may I help you?"
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")

    msg = cl.Message(content="", author="Assistant")

    res = await cl.make_async(query_engine.query)(message.content)
    # print (res)
    # for token in res.response_gen:
    #     await msg.stream_token(token)
    await cl.Message(content=res, author="Assistant").send()
   
