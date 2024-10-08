import faiss
import numpy as np

from typing import Tuple, List

import pandas as pd

from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI

API_KEY = 'YOUR KEY HERE'

def create_embeddings(df: pd.DataFrame, column_name: str, model: str) -> np.ndarray:
    """
    This function loads the OpenAI embedding model, encodes the text data in the specified column, 
    and returns a NumPy array of the embeddings.

    Args:
        df (pandas.DataFrame): The DataFrame containing the text data.
        column_name (str): The name of the column containing the text data.
        model (str): The name of the OpenAI embedding model.

    Returns:
        np.ndarray: A NumPy array containing the vector embeddings.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY, model=model)
    #encode the text data in the specified column using the sentence transformer model
    df[f"{column_name}_vector"] = df[column_name].apply(lambda x: embeddings.embed_query(x))
    #stack the encoded vectors into a NumPy array
    vectors = np.stack(df[f"{column_name}_vector"].values)
    
    return vectors

def create_index(vectors: np.ndarray, index_file_path: str) -> faiss.Index:
    """
    This function creates a FAISS index, adds the provided vectors to the index, and saves it to a file.

    Args:
        vectors (np.ndarray): A NumPy array containing the vector embeddings.
        index_file_path (str): The path to save the FAISS index file.

    Returns:
        faiss.Index: The created FAISS index.
    """
    #get the dimension of the vectors
    dimension = vectors.shape[1]
    #create a FAISS index with L2 distance metric (cosine similarity)
    index = faiss.IndexFlatL2(dimension)
    #add the vectors to the index
    index.add(vectors)
    #save the index to a file
    faiss.write_index(index, index_file_path)
    print("FAISS index is created and vectors are added to the index.")

    return index

def semantic_similarity(query: str, index: faiss.Index, model: str, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the semantic similarity between a query and a set of indexed vectors.

    Args:
        query (str): The query string.
        index (faiss.Index): The FAISS index used for searching.
        model (str): The name of the OpenAI embedding model used to create embedding.
        k (int, optional): The number of most similar vectors to retrieve. Defaults to 3.

    Returns:
        tuple: A tuple containing two arrays - D and I.
            - D (numpy.ndarray): The distances between the query vector and the indexed vectors.
            - I (numpy.ndarray): The indices of the most similar vectors in the index.
    """
    model = OpenAIEmbeddings(openai_api_key=API_KEY, model=model)
    #embed the query
    query_vector = model.embed_query(query)
    query_vector = np.array([query_vector]).astype('float32')
    #search the FAISS index
    D, I = index.search(query_vector, k)
    
    return D, I

def call_llm(query: str, responses: List[str]) -> str:
    """
    Calls the Language Model to generate a response based on the given query and list of responses.

    Args:
        query (str): The customer query.
        responses (List[str]): A list of example responses from the internal database.

    Returns:
        str: The generated response from the Language Model.
    """
    #assuming your KEY is saved in your environment variable as described in the Readme
    client = OpenAI(api_key=API_KEY)

    messages = [
        {"role": "system", "content": "You are a helpful assistant and help answer customer's query and request. Answer on the basis of provided context only."},
        {"role": "user", "content": f'''On the basis of the input customer query determine or suggest the following things about the input query: {query}:
                                          1. Urgency of the query based on the input query on a scale of 1-5 where 1 is least urgent and 5 is most urgent. Only tell me the number.
                                          2. Categorize the input query into sales, product, operations etc. Only tell me the category.
                                          3. Generate 1 best humble response to the input query which is similar to examples in the python list: {responses} from the internal database and is helpful to the customer.
                                          If the input query form customer is not clear then ask a follow up question.                  
                                      '''}
    ]
    response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0)
    
    return response.choices[0].message.content