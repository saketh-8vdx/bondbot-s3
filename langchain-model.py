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


# Load FAISS index


# Load FAISS index
faiss_index_path = 'new_index'
embeddings = CustomBedrockEmbeddings()
vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)


def retrieve_documents(query, top_k=70):
    documents = vectorstore.similarity_search(query, k=top_k)
    return [doc.page_content for doc in documents]


# def generate_response(query, documents):
#     context = "\n\n".join(documents)
#
#     system_prompt = """You are a precise and accurate assistant specializing in financial prospectus documents, particularly those related to bonds and mortgage-backed securities. Your task is to answer questions based ONLY on the information provided in the context. Follow these guidelines:
#
#     1. Provide accurate and specific answers based solely on the given context.
#     2. For numerical questions, extract and use exact figures from the context. Do not perform calculations unless explicitly asked.
#     3. If asked about ratios, percentages, or other financial metrics, quote them directly from the context.
#     4. When discussing mortgage loans, provide precise counts or amounts as stated in the document.
#     5. If the context doesn't contain the information needed to answer the question, state that the information is not available in the given context.
#     6. Do not infer, extrapolate, or add information beyond what is explicitly stated in the context.
#     7. If asked about trends or comparisons, only make such statements if they are directly supported by the context.
#     8. Use financial terminology accurately and consistently with how it's used in the context.
#
#     Remember, your primary goal is to provide accurate, context-based information without any additions or assumptions."""
#
#     prompt_template = PromptTemplate(
#         input_variables=["context", "query"],
#         template="System: {system_prompt}\n\nContext: {context}\n\nHuman: {query}\n\nAssistant:"
#     )
#
#     llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.2)
#     chain = LLMChain(llm=llm, prompt=prompt_template)
#
#     response = chain.run(context=context, query=query, system_prompt=system_prompt)
#     return response.strip()


# def analyze_query(query):
#     analysis_prompt = """Analyze the following user query and determine if it requires any specific data processing or calculation. If so, describe the type of processing needed.
#
#     User Query: {query}
#
#     Analysis:
#     1. Does this query require extracting specific numerical data? (Yes/No)
#     2. Does this query involve any calculations? (Yes/No)
#     3. If calculations are needed, what type? (e.g., summation, average, percentage)
#     4. Are there any specific financial terms or concepts that need to be identified? (List if any)
#     5. Does this query require comparing information from different parts of the document? (Yes/No)
#
#     Based on this analysis, suggest how to approach answering this query:
#     """
#
#     llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
#     response = llm.predict(analysis_prompt.format(query=query))
#     return response


def generate_response(query, documents):
    context = "\n\n".join(documents)
    print(context)
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


# @cl.on_chat_start
# def start_chat():
#     cl.user_session.set("chain", generate_response)
#
#
# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain")
#     query = message.content
#
#     # Analyze the query
#     analysis = analyze_query(query)
#     print(f"Query Analysis:\n{analysis}")
#
#     # Retrieve relevant documents
#     docs = retrieve_documents(query)
#
#     # Generate response
#     response = chain(query, docs)
#
#     await cl.Message(content=response).send()
#
#
# if __name__ == "__main__":
#     cl.run()

# query = "Number of Loans having LTV Ratio gretaer than 80 %"
# query = "Number of Loans issued based on the documentation from n from business or personal bank statements"
# query = "Number of Loans identified as investment property loans"
# query = "Number of Loans having debt-to-income ratio exceeds 40%."
query = "List the geographical areas of loan concentration and their corresponding number of loans"
# query = "give me the complete list of number of mortgage loans where Borrowers with updated credit scores below 700 include those in the credit score ranges"
# query = analyze_query(query)
# query="summarise the full document "
docs = retrieve_documents(query)

response = generate_response(query, docs)
print(response)
