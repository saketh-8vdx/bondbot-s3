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
    try:
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
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None


def load_and_chunk_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(directory_path, filename)
            try:
                print(f"Loading file: {file_path}")
                loader = UnstructuredPDFLoader(file_path, parser=PDFMinerParser())
                doc = loader.load()
                documents.extend(doc)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")

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
                "metadata": doc.metadata
            })

    print(f"Total chunks created: {len(chunked_documents)}")
    return chunked_documents


def store_embeddings_faiss(chunked_documents, faiss_index_path):
    dimensions = 1024  # Ensure this matches the embedding dimension
    faiss_index = faiss.IndexFlatL2(dimensions)

    batch_size = 100
    metadata_list = []

    for i in range(0, len(chunked_documents), batch_size):
        batch = chunked_documents[i:i + batch_size]

        ids = [str(j) for j in range(i, i + len(batch))]
        texts = [doc["chunk"] for doc in batch]
        embeddings = []

        for text in texts:
            embedding = get_embeddings(text)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                embeddings.append(np.zeros(dimensions))  # Handle errors gracefully

        embeddings_array = np.array(embeddings).astype('float32')
        faiss_index.add(embeddings_array)

        for j, doc in enumerate(batch):
            metadata = {"id": ids[j], "chunk": doc["chunk"], "metadata": doc.get("metadata", {})}
            metadata_list.append(metadata)

        print(f"Batch {i // batch_size + 1} added to FAISS index")

    faiss.write_index(faiss_index, faiss_index_path)

    with open(faiss_index_path + '_metadata.json', 'w') as f:
        json.dump(metadata_list, f)


def ingest_documents(directory_path, faiss_index_path):
    chunked_documents = load_and_chunk_documents(directory_path)
    store_embeddings_faiss(chunked_documents, faiss_index_path)


if __name__ == "__main__":
    directory_path = 'newdocuments'
    faiss_index_path = 'faiss_index1'
    start_time = time.time()
    ingest_documents(directory_path, faiss_index_path)
    end_time = time.time()

    print(f"Total time taken for ingestion: {end_time - start_time} seconds")
