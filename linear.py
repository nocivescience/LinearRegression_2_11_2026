import torch
import matplotlib.pyplot as plt

x = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
y = torch.tensor([[3.], [5.], [7.], [9.], [11.]])   # y = 2x + 1

def error(w, b):
    pred = x @ w + b
    return ((pred - y) ** 2).mean()

torch.manual_seed(0)
w = torch.randn(1, 1)
b = torch.zeros(1, 1)
e = error(w, b)

w0, b0 = w.clone(), b.clone()      # guardamos el punto de partida
historial = [e.item()]             # error en cada generación
rectas = [(w.item(), b.item())]    # (w, b) en cada generación

for gen in range(1, 201):
    for _ in range(20):
        w2 = w + torch.randn(1, 1) * 0.1
        b2 = b + torch.randn(1, 1) * 0.1
        e2 = error(w2, b2)
        if e2 < e:
            w, b, e = w2, b2, e2
    historial.append(e.item())
    rectas.append((w.item(), b.item()))

# ---------- Gráficos ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Izquierda: datos y rectas
ax1.scatter(x, y, color="black", zorder=3, label="datos")
xs = torch.linspace(0, 6, 50)
for i in (0, 5, 20, 200):                       # algunas generaciones
    wi, bi = rectas[i]
    ax1.plot(xs, wi * xs + bi, alpha=0.7, label=f"gen {i}")
ax1.set_title("La recta se acerca a los datos")
ax1.set_xlabel("x"); ax1.set_ylabel("y")
ax1.legend()

# Derecha: pérdida vs generación
ax2.plot(historial)
ax2.set_yscale("log")                           # escala log para ver bien la caída
ax2.set_title("La pérdida baja con las generaciones")
ax2.set_xlabel("generación"); ax2.set_ylabel("error")

plt.tight_layout()
plt.show()