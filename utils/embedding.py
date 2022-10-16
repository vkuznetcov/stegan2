def additional_embedding(f, alpha, beta, omega):
    return f + alpha * beta * omega


def multiplication_embedding(f, alpha, beta, omega):
    return f * (1 + alpha * beta * omega)
