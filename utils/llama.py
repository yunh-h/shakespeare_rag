from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

generator_pipeline = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    device=device, 
    pad_token_id=tokenizer.eos_token_id
)

def generate_answer(query, retrieved_passages):
    """
    Generate an answer using Llama models.
    """
    system_prompt = (
        "You are a helpful assistant that answers literary questions about William Shakespeare's work."
        "Use only the passages provided to answer the question. Do not repeat the input passages and the questions. Do not make up facts. Answer accurately and concisely.\n\n"
    )

    question_part = f"User Question: {query}\n\n"

    passages_part = "Relevant Passages:\n"
    for idx, passage_text in enumerate(retrieved_passages):
        passages_part += f"[Passage #{idx+1}]: {passage_text}\n"

    answer_prompt = "\nAnswer concisely and accurately:\n"
    combined_prompt = system_prompt + question_part + passages_part + answer_prompt
    outputs = generator_pipeline(combined_prompt, max_new_tokens=500, do_sample=True)
    output = outputs[0]["generated_text"]
    answer = output.replace(combined_prompt, "").strip()

    return answer