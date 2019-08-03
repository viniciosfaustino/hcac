import numpy as np

def troca(a: np.ndarray):
    a = a*-1
    print(a)

def dicto(b: dict):
    b[21] = "deu ruim"

if __name__ == '__main__':
    a = np.arange(10)
    c = {1:2}
    print(a)
    troca(a)
    print(a)

    print(c)
    dicto(c)
    print(c)