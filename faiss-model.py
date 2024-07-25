import openai
import faiss
import boto3
import json
import numpy as np
import os
import chainlit as cl
import time
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'  # Replace with your AWS region
)

# Load FAISS index
faiss_index_path = 'faiss_index'
faiss_index = faiss.read_index(faiss_index_path)

# Load metadata
metadata_path = 'faiss_index_metadata.json'
with open(metadata_path, 'r') as f:
    metadata = [json.loads(line) for line in f]


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


def retrieve_documents(query, top_k=10):
    # Assuming you have a method to get embeddings for the query
    query_embedding = get_embeddings(query)
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)

    distances, indices = faiss_index.search(query_embedding, top_k)
    documents = [metadata[i]['chunk'] for i in indices[0]]
    print(len(documents))
    return documents

def generate_response(query, documents):
    context = "\n\n".join(documents)
    print(context)
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system",
             "content": "You are a precise and accurate assistant answering questions related to the bonds present in "
                        "the prospectus document. Respond to the user's question based ONLY on "
                        "the information provided in the context. Do not add, infer, or generate any information "
                        "beyond what is explicitly stated. For questions that involves computation of number of "
                        "mortgage loans extract information from the context and answer it accordingly, provide exact "
                        "numerical "
                        "answers as given in the context. If the "
                        "context doesn't "
                        "contain the information needed "
                        "to answer the question, do not generate any response."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        max_tokens=1000,
        temperature=0.2,
        top_p=0.8,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    return response.choices[0].message['content'].strip()

#

# async def generate_response(query, documents):
#     context = "\n\n".join(documents)
#     print(context)
#     max_context_tokens = 6000  # Adjust as necessary
#     max_response_tokens = 1000  # Set high to avoid truncation
#     context_chunks = []
#
#     while len(context) > 0:
#         chunk = context[:max_context_tokens]
#         context_chunks.append(chunk)
#         context = context[max_context_tokens:]
#
#     final_response = ""
#
#     for chunk in context_chunks:
#         # Call the GPT-4 Turbo model
#         response = openai.ChatCompletion.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system",
#                  "content": "You are a precise and accurate assistant. Respond to the user's question based ONLY on "
#                             "the information provided in the context. Do not add, infer, or generate any information "
#                             "beyond what is explicitly stated. For quantitative questions, provide exact numerical "
#                             "answers as given in the context."},
#                 {"role": "user", "content": f"Context: {chunk}\n\nQuestion: {query}\n\nAnswer:"}
#             ],
#
#             max_tokens=max_response_tokens,
#             temperature=0.3,
#             top_p=0.7,
#             frequency_penalty=0.0,
#             presence_penalty=0.0
#         )
#         print(response.choices[0].message['content'].strip())
#
#         final_response += response.choices[0].message['content'].strip() + "\n\n"
#
#     return final_response

#
query = 'give me the complete details of the number of loans issued based on the bank statements'
doc = retrieve_documents(query)
res = generate_response(query, doc)
print('-----------------------------------------------------------------------------------------------------------------------------------------------')
print(res)




# @cl.on_chat_start
# async def start():
#     await cl.Message(content="Welcome to Band-bot!!").send()
#
#
# @cl.on_message
# async def main(message: cl.Message):
#     query = message.content
#     print("received prompt")
#     start = time.time()
#     documents = await retrieve_documents(query)
#     print("documents retrived")
#     response = await generate_response(query, documents)
#     end = time.time()
#     response += f"\n\n\n Time taken= {end - start} "
#
#     print("RESPONSE\n\n\n")
#     print(response)
#     await cl.Message(content=response).send()
#
#
# if __name__ == "__main__":
#     cl.run()
