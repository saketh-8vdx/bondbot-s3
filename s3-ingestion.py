import time
import os
import json
import boto3
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.parsers.pdf import PDFMinerParser
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from langchain.schema import Document
from botocore.exceptions import ClientError
import requests
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

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


def get_document_from_s3(bucket_name, s3_key):
    s3_client = boto3.client('s3')

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        document_content = response['Body'].read()
        return document_content

    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"The object with key '{s3_key}' does not exist in the bucket '{bucket_name}'.")
        elif e.response['Error']['Code'] == 'NoSuchBucket':
            print(f"The bucket '{bucket_name}' does not exist.")
        else:
            print(f"An error occurred: {e}")

        return None


def load_and_chunk_documents_from_s3(fetch_bucket_name, s3_key):
    document_content = get_document_from_s3(fetch_bucket_name, s3_key)

    if document_content is None:
        return []

    # Save the document content to a temporary file
    temp_file_path = '/tmp/temp_document.pdf'
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(document_content)

    # Load the document using UnstructuredPDFLoader
    # loader = UnstructuredPDFLoader(temp_file_path, parser=PDFMinerParser())
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()

    # Clean up the temporary file
    os.remove(temp_file_path)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )

    chunked_documents = text_splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunked_documents)}")

    return chunked_documents


def upload_files_to_s3(store_bucket_name, file_paths, prospectus_key):
    # Initialize the S3 client
    s3_client = boto3.client('s3')
    uploaded_keys = []
    for file_path in file_paths:
        try:
            # Extract the file name from the file path
            file_name = f"{prospectus_key}/{os.path.basename(file_path)}"

            # Upload the file
            s3_client.upload_file(file_path, store_bucket_name, file_name)
            print(f"Successfully uploaded {file_name} to {store_bucket_name}")
            uploaded_keys.append(file_name)

        except FileNotFoundError:
            print(f"The file {file_path} was not found")
        except NoCredentialsError:
            print("Credentials not available")
        except PartialCredentialsError:
            print("Incomplete credentials provided")
        except Exception as e:
            print(f"An error occurred: {e}")
    return uploaded_keys


def post_to_api(keys, api, prospectus_key):
    # Prepare the request body
    payload = {
        prospectus_key: keys
    }

    try:
        # Send POST request to the API
        response = requests.post(api, json=payload)

        # Check if the request was successful
        if response.status_code == 200:
            print("Successfully posted to API")
            print("API Response:", response.json())
        else:
            print(f"Failed to post to API. Status code: {response.status_code}")
            print("API Response:", response.text)

        return response

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while posting to API: {e}")
        return None


def store_embeddings_faiss_and_upload_to_s3(chunked_documents, store_bucket_name, post_api, prospectus_key,
                                            temp_dir='/tmp/faiss_index'):
    embeddings = CustomBedrockEmbeddings()

    # Create FAISS index
    vectorstore = FAISS.from_documents(chunked_documents, embeddings)

    # Save the FAISS index locally (temporarily)
    os.makedirs(temp_dir, exist_ok=True)
    vectorstore.save_local(temp_dir)

    # Get the file names
    pkl_file = next(f for f in os.listdir(temp_dir) if f.endswith('.pkl'))
    faiss_file = next(f for f in os.listdir(temp_dir) if f.endswith('.faiss'))
    file_paths = [os.path.join(temp_dir, pkl_file), os.path.join(temp_dir, faiss_file)]

    # Upload files to S3
    keys_list = upload_files_to_s3(store_bucket_name, file_paths, prospectus_key)

    res = post_to_api(keys_list, post_api, prospectus_key)

    return res


def get_s3_keys_from_api(api_url):
    try:
        # Send GET request to the API
        response = requests.get(api_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Extract the s3Keys list
            s3_keys = data.get('data', {}).get('s3Keys', [])

            return s3_keys
        else:
            print(f"Failed to get data from API. Status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the API request: {e}")
        return None


fetch_bucket_name = 'dev-securities-prospectus'
get_api = "https://dev-api.ventureinsights.ai/jobs/security-master/get-all-prospectus/BOND"
post_api= 'https://dev-api.ventureinsights.ai/jobs/security-master/update-embeddings-for-prospectus'
store_bucket_name = "dev-8vdx-securities-embeddings"

prospectus_list = get_s3_keys_from_api(get_api)

for prospectus_key in prospectus_list:
    print(prospectus_key)
    chunked_documents = load_and_chunk_documents_from_s3(fetch_bucket_name, prospectus_key)

    # if chunked_documents:
    #     print(f"Successfully chunked the document into {len(chunked_documents)} chunks.")
    #     res = store_embeddings_faiss_and_upload_to_s3(chunked_documents, store_bucket_name, post_api,prospectus_key)
    #     if res:
    #         print("done")
    #     else:
    #         print("I am Sorry")

    #
    # else:
    #     print("Failed to load and chunk the document.")








