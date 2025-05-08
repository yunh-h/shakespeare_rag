import openai

# Set your OpenAI API key
client = openai.OpenAI(api_key="")

def generate_answer_with_gpt(query, retrieved_passages):
    """
    Generate an answer using OpenAI GPT models.
    """
    system_prompt = (
        "You are a helpful assistant that answers literary questions accurately and concisely. "
        "Use only the passages provided to answer the question. Do not repeat the passages or make up facts.\n\n"
    )

    question_part = f"User Question: {query}\n\n"
    passages_part = "Relevant Passages:\n"

    for idx, passage_text in enumerate(retrieved_passages):
        passages_part += f"[Passage #{idx+1}]: {passage_text}\n"

    answer_prompt = "\nAnswer concisely and accurately:\n"
    combined_prompt = system_prompt + question_part + passages_part + answer_prompt

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": combined_prompt}
        ],
        max_tokens=100,
        temperature=0.5
    )

    answer = response.choices[0].message.content

    return answer