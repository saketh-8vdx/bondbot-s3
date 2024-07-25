import openai
import boto3
import json
import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings

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


faiss_index_path = 'newfaiss_index'
embeddings = CustomBedrockEmbeddings()
vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)


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



query = ""

docs = retrieve_documents(query)

response = generate_response(query, docs)
print(response)
