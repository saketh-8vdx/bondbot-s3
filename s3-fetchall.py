

# Function to load a FAISS index from .pkl and .faiss files
def load_faiss_index(pkl_file, faiss_file):
    index = FAISS.load_local(pkl_file, embeddings)
    index.load_local(faiss_file)
    return index


# Function to search across multiple FAISS indexes
def search_multiple_indexes(query, index_files, k=5):
    all_results = []

    for pkl_file, faiss_file in index_files:
        index = load_faiss_index(pkl_file, faiss_file)
        results = index.similarity_search_with_score(query, k=k)
        all_results.extend(results)

    # Sort all results by score (assuming lower score is better)
    all_results.sort(key=lambda x: x[1])

    # Return top k results
    return all_results[:k]


# List of tuples containing paths to your .pkl and .faiss files
index_files = [
    ("document1/index.pkl", "document1/index.faiss"),
    ("document2/index.pkl", "document2/index.faiss"),
    ("document3/index.pkl", "document3/index.faiss"),
    # Add more file pairs as needed
]

# Your query
query = "Your search query here"

# Perform the search
results = search_multiple_indexes(query, index_files)

# Print the results
for doc, score in results:
    print(f"Document: {doc.metadata['source']}, Score: {score}")
    print(f"Content: {doc.page_content[:100]}...")  # Print first 100 characters
    print()