import time
import os
import json
import boto3
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders.parsers.pdf import PDFMinerParser

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)


def get_embeddings(text, dimensions=1024, normalize=True):
    body = json.dumps({
        "inputText": text,
        "dimensions": dimensions,
        "normalize": normalize
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
    documents = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            print(file_path)
            # Load each PDF file
            loader = UnstructuredPDFLoader(file_path, parser=PDFMinerParser())
            doc = loader.load()
            documents.extend(doc)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    chunked_documents = []
    for doc in documents:
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            chunked_documents.append({
                "chunk": chunk,
                # "metadata": doc.metadata
            })

    print(len(chunked_documents))

    return chunked_documents


def store_embeddings_faiss(chunked_documents, faiss_index_path):
    # Initialize FAISS index
    dimensions = 1024  # Make sure this matches the embedding dimension
    faiss_index = faiss.IndexFlatL2(dimensions)

    # Batch process documents
    batch_size = 100
    for i in range(0, len(chunked_documents), batch_size):
        batch = chunked_documents[i:i + batch_size]

        ids = [str(j) for j in range(i, i + len(batch))]
        texts = [doc["chunk"] for doc in batch]
        embeddings = [get_embeddings(text) for text in texts]

        # Convert embeddings to numpy array and add to FAISS index
        embeddings_array = np.array(embeddings).astype('float32')
        faiss_index.add(embeddings_array)

        # Save the metadata
        with open(faiss_index_path + '_metadata.json', 'a') as f:
            for j, doc in enumerate(batch):
                metadata = {"id": ids[j], "chunk": doc["chunk"]}
                json.dump(metadata, f)
                f.write("\n")

        print(f"Batch {i // batch_size + 1} added to FAISS index")

    # Save the FAISS index
    faiss.write_index(faiss_index, faiss_index_path)


def ingest_documents(directory_path, faiss_index_path):
    chunked_documents = load_and_chunk_documents(directory_path)
    store_embeddings_faiss(chunked_documents, faiss_index_path)


if __name__ == "__main__":
    directory_path = 'newdocuments'
    faiss_index_path = 'faiss_index'
    start_time = time.time()
    ingest_documents(directory_path, faiss_index_path)
    end_time = time.time()

    print(f"Total time taken for ingestion: {end_time - start_time} seconds")
