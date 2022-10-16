def additional_embedding(f, beta, omega, alpha=1):
    return f + alpha * beta * omega


def multiplication_embedding(f, beta, omega, alpha=1):
    return f * (1 + alpha * beta * omega)
