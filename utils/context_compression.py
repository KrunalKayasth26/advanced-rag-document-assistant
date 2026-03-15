def compress_context(client, query, docs):

    compressed = []

    for doc in docs:

        prompt = f"""
Extract only the sentences relevant to the question.

Question:
{query}

Text:
{doc.page_content}

Relevant sentences:
"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Extract relevant sentences only."},
                {"role": "user", "content": prompt}
            ]
        )

        text = response.choices[0].message.content.strip()

        if text:
            compressed.append(text)

    return "\n".join(compressed)