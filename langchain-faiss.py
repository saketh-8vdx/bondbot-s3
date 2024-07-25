import time
import os
import json
import boto3
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders.parsers.pdf import PDFMinerParser
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from langchain.schema import Document

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)


class CustomBedrockEmbeddings(BedrockEmbeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        body = json.dumps({
            "inputText": text,
            "dimensions": 1024,
            "normalize": True
        })

        response = bedrock.invoke_model(
            body=body,
            modelId='amazon.titan-embed-text-v2:0',
            contentType='application/json',
            accept='application/json'
        )

        response_body = json.loads(response['body'].read())
        return response_body['embedding']


def load_and_chunk_documents(directory_path):
    loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader,
                             loader_kwargs={"parser": PDFMinerParser()})
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )

    chunked_documents = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunked_documents)}")

    return chunked_documents


def store_embeddings_faiss(chunked_documents, faiss_index_path):
    embeddings = CustomBedrockEmbeddings()

    # Create FAISS index
    vectorstore = FAISS.from_documents(chunked_documents, embeddings)

    # Save the FAISS index
    vectorstore.save_local(faiss_index_path)


def ingest_documents(directory_path, faiss_index_path):
    chunked_documents = load_and_chunk_documents(directory_path)
    store_embeddings_faiss(chunked_documents, faiss_index_path)


if __name__ == "__main__":
    directory_path = 'newdocuments'
    faiss_index_path = 'new'
    start_time = time.time()
    ingest_documents(directory_path, faiss_index_path)
    end_time = time.time()

    print(f"Total time taken for ingestion: {end_time - start_time} seconds")