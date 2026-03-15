def choose_k(query):

    length = len(query.split())

    if length <= 3:
        return 3

    if length <= 8:
        return 5

    return 8