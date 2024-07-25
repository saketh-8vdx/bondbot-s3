import requests
import openai
import boto3
import json
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
import shutil

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

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


def get_file_details_from_api(api_url, s3_key):
    try:
        # Prepare the parameters for the GET request

        full_url = api_url.replace(':s3Key', s3_key)

        # Send GET request to the API with parameters
        response = requests.get(full_url)
        # print(response.text)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            file_details = response.json()

            return file_details
        else:
            print(f"Failed to get data from API. Status code: {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while making the API request: {e}")
        return None


# def download_files_from_s3(bucket_name, s3_keys, download_dir):
#     s3_client = boto3.client('s3')
#     downloaded_files = []
#
#     for s3_key in s3_keys:
#         try:
#             # Define the local file path
#             local_file_path = os.path.join(download_dir, os.path.basename(s3_key))
#
#             # Download the file from S3
#             s3_client.download_file(bucket_name, s3_key, local_file_path)
#             print(f"Successfully downloaded {s3_key} to {local_file_path}")
#
#             # Add the local file path to the list
#             downloaded_files.append(local_file_path)
#
#         except Exception as e:
#             print(f"An error occurred while downloading {s3_key}: {e}")
#
#     return downloaded_files

#


# def download_files_from_s3(bucket_name, s3_keys, download_dir):
#     s3_client = boto3.client('s3')
#     downloaded_files = []
#
#     for s3_key in s3_keys:
#         try:
#             # Define the local file path
#             local_file_path = os.path.join(download_dir, os.path.basename(s3_key))
#             print(local_file_path)
#
#             # Download the file from S3
#             s3_client.download_file(bucket_name, s3_key, local_file_path)
#             print(f"Successfully downloaded {s3_key} to {local_file_path}")
#
#             # Add the local file path to the list
#             downloaded_files.append(local_file_path)
#
#         except FileNotFoundError:
#             print(f"The file {s3_key} was not found in the bucket {bucket_name}.")
#         except NoCredentialsError:
#             print("Credentials not available.")
#         except PartialCredentialsError:
#             print("Incomplete credentials provided.")
#         except ClientError as e:
#             error_code = e.response['Error']['Code']
#             if error_code == '404':
#                 print(f"The object {s3_key} does not exist in the bucket {bucket_name}.")
#             else:
#                 print(f"An error occurred: {e.response['Error']['Message']}")
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}")
#
#     return downloaded_files


def newdownload_files_from_s3(bucket_name, s3_keys, download_dir):
    s3_client = boto3.client('s3')
    downloaded_files = []

    for s3_key in s3_keys:
        try:
            # Define the local file path
            local_file_path = os.path.join(download_dir, s3_key)
            print(local_file_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download the file from S3
            s3_client.download_file(bucket_name, s3_key, local_file_path)
            print(f"Successfully downloaded {s3_key} to {local_file_path}")

            # Add the local file path to the list
            downloaded_files.append(local_file_path)

        except FileNotFoundError:
            print(f"The file {s3_key} was not found in the bucket {bucket_name}.")
        except NoCredentialsError:
            print("Credentials not available.")
        except PartialCredentialsError:
            print("Incomplete credentials provided.")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                print(f"The object {s3_key} does not exist in the bucket {bucket_name}.")
            else:
                print(f"An error occurred: {e.response['Error']['Message']}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    return downloaded_files


# def download_files(bucket_name, s3_keys, download_dir):
#     s3_client = boto3.client('s3')
#     downloaded_files = []
#
#     for s3_key in s3_keys:
#         try:
#             # Define the local file path
#             local_file_path = os.path.join(download_dir, s3_key['embeddingsS3Keys'][1])
#             print(local_file_path)
#
#             # Download the file from S3
#             s3_client.download_file(bucket_name, s3_key['embeddingsS3Keys'][1], local_file_path)
#             print(f"Successfully downloaded {s3_key['embeddingsS3Keys'][1]} to {local_file_path}")
#
#             # Add the local file path to the list
#             downloaded_files.append(local_file_path)
#
#         except FileNotFoundError:
#             print(f"The file {s3_key['embeddingsS3Keys'][1]} was not found in the bucket {bucket_name}.")
#         except NoCredentialsError:
#             print("Credentials not available.")
#         except PartialCredentialsError:
#             print("Incomplete credentials provided.")
#         except ClientError as e:
#             error_code = e.response['Error']['Code']
#             if error_code == '404':
#                 print(f"The object {s3_key} does not exist in the bucket {bucket_name}.")
#             else:
#                 print(f"An error occurred: {e.response['Error']['Message']}")
#         except Exception as e:
#             print(f"An unexpected error occurred: {e}")
#
#     return downloaded_files


def retrieve_documents(query, top_k=55):
    documents = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in documents]


def generate_response(query, documents):
    context = "\n\n".join(documents)
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system",
             "content": """You are a precise and accurate assistant specializing in prospectus documents, particularly those related to bonds and mortgage-backed securities. Your responses should strictly adhere to the following guidelines:

             1. For theoretical or summary questions:
                - Provide a concise and accurate summary based solely on the information in the given context.
                - Do not add any information beyond what is explicitly stated in the context.
                - If the context doesn't contain relevant information, state that the information is not available.

             2. For computational or quantitative questions:
                - Extract exact numerical data, ratios, or percentages from the context.
                - Provide precise answers using only the figures given in the context.
                - For questions about mortgage loans or bond quantities, use the exact numbers stated in the context.
                -For questions related to counting the number of loans having som ecategory first extract all loans satisfying the condition and tehn count and return the detailed response
               -For the questions like number of lons issued based on some category, extract information regarding the category from the context and then return the detailed response

             3. General guidelines:
                - Do not infer, extrapolate, or add any information beyond what is explicitly stated in the context.
                - Use financial terminology accurately and consistently with how it's used in the context.


             Remember, your primary goal is to provide accurate, context-based information without any additions, assumptions, or hallucinations.Dont forget to return the souce by explicitly mentioning from the context from whetre you have generated the response"""},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        max_tokens=1000,
        temperature=0.2,
        top_p=0.8,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response.choices[0].message['content'].strip()


def cleanup_directory(directory):
    try:
        shutil.rmtree(directory)
        print(f"Successfully removed directory: {directory}")
    except Exception as e:
        print(f"Error removing directory {directory}: {e}")


# api_url = "https://dev-api.ventureinsights.ai/jobs/security-master/embeddings-using-s3-key/:s3Key"
#
# s3_key = "Copy of 2000Aug23-24.pdf"
#
# file_details = get_file_details_from_api(api_url, s3_key)
# embedding_files = file_details['data'][0]['embeddingsS3Keys']
# bucket_name = 'dev-8vdx-securities-embeddings'
# download_dir = 'tmp'
# os.makedirs(download_dir, exist_ok=True)
#
# downloaded_files = download_files_from_s3(bucket_name, embedding_files, download_dir)
#
# embeddings = CustomBedrockEmbeddings()
# vectorstore = FAISS.load_local(download_dir, embeddings, allow_dangerous_deserialization=True)
#
# query = "list all loans "
#
# docs = retrieve_documents(query)
#
# response = generate_response(query, docs)
# print(response)
# cleanup_directory(download_dir)
















# def load_faiss_index(pkl_file, faiss_file):
#     index = FAISS.load_local(pkl_file, embeddings)
#     index.load_local(faiss_file)
#     return index
#
#
# def find_index_files(directory):
#     index_files = []
#     for root, dirs, files in os.walk(directory):
#         pkl_file = None
#         faiss_file = None
#         for file in files:
#             if file.endswith('.pkl'):
#                 pkl_file = os.path.join(root, file)
#             elif file.endswith('.faiss'):
#                 faiss_file = os.path.join(root, file)
#         if pkl_file and faiss_file:
#             index_files.append((pkl_file, faiss_file))
#     return index_files
#
#
# def search_multiple_indexes(query, directory, k=5):
#     all_results = []
#     index_files = find_index_files(directory)
#
#     for pkl_file, faiss_file in index_files:
#         try:
#             index = load_faiss_index(pkl_file, faiss_file)
#             results = index.similarity_search_with_score(query, k=k)
#             all_results.extend(results)
#         except Exception as e:
#             print(f"Error processing {pkl_file} and {faiss_file}: {str(e)}")
#
#     # Sort all results by score (assuming lower score is better)
#     all_results.sort(key=lambda x: x[1])
#
#     # Return top k results
#     return all_results[:k]
#
#
api_url = "https://dev-api.ventureinsights.ai/jobs/security-master/embeddings-using-s3-key/all"
response = requests.get(api_url)
file_details = response.json()
list = file_details['data']
bucket_name = 'dev-8vdx-securities-embeddings'
download_dir = 'folder'
os.makedirs(download_dir, exist_ok=True)

embeddings = CustomBedrockEmbeddings()



for file in list:
    print(file['embeddingsS3Keys'][1])
    newdownload_files_from_s3(bucket_name, file['embeddingsS3Keys'], download_dir)
















def load_faiss_index(directory):
    # Load the FAISS index with the allow_dangerous_deserialization parameter
    return FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)


def find_index_directories(directory):
    index_directories = []
    for root, dirs, files in os.walk(directory):
        if any(file.endswith('.pkl') for file in files) and any(file.endswith('.faiss') for file in files):
            index_directories.append(root)
    return index_directories


def search_multiple_indexes(query, directory, k=55):
    all_results = []
    index_directories = find_index_directories(directory)

    for index_directory in index_directories:
        try:
            index = load_faiss_index(index_directory)
            results = index.similarity_search_with_score(query, k=k)
            all_results.extend(results)
        except Exception as e:
            print(f"Error processing index in {index_directory}: {str(e)}")

    # Sort all results by score (assuming lower score is better)
    all_results.sort(key=lambda x: x[1])

    # Return top k results
    return all_results[:k]


# Example usage
directory = "folder"
query = "List all CUSIP Numbers"
results = search_multiple_indexes(query, directory)

# Print the results
# for doc, score in results:
#     print(f"Document: {doc.metadata.get('source', 'Unknown')}, Score: {score}")
#     print(f"Content: {doc.page_content[:100]}...")  # Print first 100 characters
#     print()
#     break

# If you want to return just the page contents
page_contents = [doc.page_content for doc, _ in results]
print(len(page_contents))