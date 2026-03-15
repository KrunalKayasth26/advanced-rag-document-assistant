def expand_query(client, query):

    prompt = f"""
Generate 3 different search queries that mean the same as:

{query}

Return only the queries.
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You generate search queries."},
            {"role": "user", "content": prompt}
        ]
    )

    text = response.choices[0].message.content

    queries = [q.strip("- ").strip() for q in text.split("\n") if q.strip()]

    queries.insert(0, query)

    return queries[:4]