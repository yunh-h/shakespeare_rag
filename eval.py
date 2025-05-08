import json
from utils.retrieve import retrieve_passages_with_keywords, embedder, index, chunks_with_metadata
from utils.openai import generate_answer_with_gpt
from utils.llama import generator_pipeline

def retrieve_passages_eval(query, top_k=5):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)

    # Retrieve actual text
    results = []
    for i, idx in enumerate(indices[0]):
        result = chunks_with_metadata[idx]
        results.append({
            "rank": i + 1,
            "id": result["id"],
            "name": result["name"],
            "contents": result["contents"],
            "distance": distances[0][i]
        })
        
    print(f"Query: {query}")
    print(f"Top {top_k} results:")
    for res in results:
        print(f"Rank: {res['rank']}, ID: {res['id']}, Name: {res['name']}, Distance: {res['distance']}")
    print()

    return results

def compute_metrics(dataset, retriever, top_k=5):
    """
    Compute Recall@k and MRR for the retrieval system.

    Args:
        dataset (list): List of Q&A pairs (questions and answers).
        retriever (function): Function to retrieve passages.
        top_k (int): Number of top passages to retrieve.

    Returns:
        dict: Metrics (Recall@k, MRR).
    """
    total_questions = len(dataset)
    recall_at_k = 0
    reciprocal_ranks = []

    for qa in dataset:
        question = qa["question"]
        ground_truth = qa["answer"]

        retrieved_passages = retriever(question, top_k=top_k)

        # Check if the ground truth is in the top-k results
        found = False
        for rank, passage in enumerate(retrieved_passages, start=1):
            if ground_truth.lower() in passage["name"].lower():
                recall_at_k += 1
                reciprocal_ranks.append(1 / rank)
                found = True
                break

        if not found:
            reciprocal_ranks.append(0)

    # Compute metrics
    recall_at_k /= total_questions
    mrr = sum(reciprocal_ranks) / total_questions

    return {
        "Recall@k": recall_at_k,
        "MRR": mrr
    }

def retriever(query, top_k=5):
    return retrieve_passages_with_keywords(query, top_k=top_k)

def evaluate_and_print(model, dataset_path, top_k=5):
    """
    Evaluate the model on a dataset and print outputs for each question.

    Args:
        model: The language model or pipeline for generating answers.
        dataset_path (str): Path to the QA dataset (JSON file).
        top_k (int): Number of top passages to retrieve.
    """
    with open(dataset_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    for qa in dataset:
        question = qa["question"]

        # Retrieve relevant passages
        retrieved_passages = retriever(question, top_k=top_k)
        # retrieved_texts = [passage["contents"] for passage in retrieved_passages]
        retrieved_ids = [passage["id"] for passage in retrieved_passages]

        if retrieved_ids:
            # Combine retrieved passages into a single context
            context = " ".join(retrieved_ids)
            input_text = f"Question: {question}\nContext: {context}\nAnswer:"
        else:
            context = "No context retrieved."
            input_text = f"Question: {question}\nAnswer:"

        # Generate the answer
        generated = model(input_text)
        generated_answer = generated[0]["generated_text"].strip()

        print(f"Question: {question}")
        print(f"Context: {context}")
        print(f"Generated Answer: {generated_answer}")
        print("-" * 80)

if __name__ == "__main__":
    with open("eval/line_loc.json", "r", encoding="utf-8") as file:
        dataset = json.load(file)
    
    metrics = compute_metrics(dataset, retrieve_passages_eval, top_k=5)

    print(f"Recall@5: {metrics['Recall@k']:.4f}")
    print(f"MRR: {metrics['MRR']:.4f}")
    print("-" * 80)

    print("Evaluating LLaMA...")
    evaluate_and_print(
        model=generator_pipeline,
        dataset_path="eval/understanding.json",
        top_k=5
    )
    print("-" * 80)
    
    print("Evaluating GPT...")
    def gpt_model(input_text):
        query = input_text.split("Question:")[1].split("\n")[0].strip()
        retrieved_passages = retriever(query, top_k=5)
        return [{"generated_text": generate_answer_with_gpt(query, retrieved_passages)}]

    evaluate_and_print(
        model=gpt_model,
        dataset_path="eval/understanding.json",
        top_k=5
    )