import torch

x = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
y = torch.tensor([[3.], [5.], [7.], [9.], [11.]])   # y = 2x + 1

def error(w, b):
    pred = x @ w + b                  # la "neurona"
    return ((pred - y) ** 2).mean()   # qué tan lejos quedó de la realidad

torch.manual_seed(0)
w = torch.randn(1, 1)                 # perilla 1 al azar
b = torch.zeros(1, 1)                 # perilla 2 en cero
e = error(w, b)

for gen in range(1, 501):
    for _ in range(20):               # 20 hijos por generación
        w2 = w + torch.randn(1, 1) * 0.1    # mutar un poquito
        b2 = b + torch.randn(1, 1) * 0.1
        e2 = error(w2, b2)
        if e2 < e:                    # si es mejor, nos quedamos con él
            w, b, e = w2, b2, e2
    if gen % 100 == 0 or gen == 1:
        print(f"gen {gen:3d}  w={w.item():.3f}  b={b.item():.3f}  error={e.item():.5f}")