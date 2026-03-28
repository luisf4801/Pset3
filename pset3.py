# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from numpy.polynomial.hermite import hermgauss
import time
import os

# ── Pasta de output ───────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
out_dir    = os.path.join(script_dir, 'output')
os.makedirs(out_dir, exist_ok=True)

def savefig(name):
    plt.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches='tight')
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# QUESTÃO 1
# ══════════════════════════════════════════════════════════════════════════════
def f_q1(x):
    return x * np.sin(5 * x)

x_real = 9.74304869
f_real = f_q1(x_real)

# ── Grid Search ───────────────────────────────────────────────────────────────
x_grid    = np.linspace(0, 10, 100000)
f_grid    = f_q1(x_grid)
x_min_grid = x_grid[np.argmin(f_grid)]
f_min_grid = f_grid.min()
erro_grid  = abs(f_real - f_min_grid)

print(f"Grid Search → x* = {x_min_grid:.6f},  f(x*) = {f_min_grid:.6f},  erro = {erro_grid:.2e}")
print("Grid Search achou o mínimo" if erro_grid < 1e-4 else "Grid Search não achou o mínimo")

x = np.linspace(0, 10, 1000)
plt.figure(figsize=(10, 5))
plt.plot(x, f_q1(x), label='f(x) = x·sin(5x)', color='steelblue')
plt.scatter(x_real,     f_real,     color='green', zorder=5, s=100,
            label=f'Mínimo real ({x_real:.5f}, {f_real:.5f})')
plt.scatter(x_min_grid, f_min_grid, color='red',   zorder=5, s=100, marker='x',
            label=f'Grid Search ({x_min_grid:.5f}, {f_min_grid:.5f})')
plt.xlabel('x'); plt.ylabel('f(x)')
plt.title('f(x) = x·sin(5x) — Mínimo real vs Grid Search')
plt.legend(); plt.grid(True); plt.tight_layout()
savefig('q1_grid_search.png')

# ── Brent ─────────────────────────────────────────────────────────────────────
def brent_method(f, bounds, tol=1e-4):
    res = minimize_scalar(f, bounds=bounds, method='bounded',
                          options={'xatol': tol})
    return res.x, res.fun

x_min_brent, f_min_brent = brent_method(f_q1, bounds=(0, 10))
erro_brent = abs(f_real - f_min_brent)

print(f"Brent → x* = {x_min_brent:.6f},  f(x*) = {f_min_brent:.6f},  erro = {erro_brent:.2e}")
print("Brent achou o mínimo" if erro_brent < 1e-4 else "Brent não achou o mínimo")

plt.figure(figsize=(10, 5))
plt.plot(x, f_q1(x), label='f(x) = x·sin(5x)', color='steelblue')
plt.scatter(x_real,      f_real,      color='green', zorder=5, s=100,
            label=f'Mínimo real ({x_real:.5f}, {f_real:.5f})')
plt.scatter(x_min_brent, f_min_brent, color='red',   zorder=5, s=100, marker='x',
            label=f'Brent ({x_min_brent:.5f}, {f_min_brent:.5f})')
plt.xlabel('x'); plt.ylabel('f(x)')
plt.title('f(x) = x·sin(5x) — Mínimo real vs Brent')
plt.legend(); plt.grid(True); plt.tight_layout()
savefig('q1_brent.png')

# ── Nelder-Mead ───────────────────────────────────────────────────────────────
def nelder_mead_opt(objetivo, x0, tol=1e-4):
    res = minimize(objetivo, x0, method='nelder-mead',
                   options={'xatol': tol, 'disp': False})
    return res.x, res.fun

x_min_nm, f_min_nm = nelder_mead_opt(f_q1, x0=0)
erro_nm = abs(f_real - f_min_nm)

print(f"Nelder-Mead → x* = {x_min_nm[0]:.6f},  f(x*) = {f_min_nm:.6f},  erro = {erro_nm:.2e}")
print("Nelder-Mead achou o mínimo global!" if erro_nm < 1e-4 else "Nelder-Mead parou em um mínimo local.")

plt.figure(figsize=(10, 5))
plt.plot(x, f_q1(x), label='f(x) = x·sin(5x)', color='steelblue')
plt.scatter(x_real,   f_real,   color='green', zorder=5, s=100,
            label=f'Mínimo real ({x_real:.5f})')
plt.scatter(x_min_nm, f_min_nm, color='red',   zorder=6, s=100, marker='x',
            label=f'Nelder-Mead ({x_min_nm[0]:.5f})')
plt.xlabel('x'); plt.ylabel('f(x)')
plt.title('f(x) = x·sin(5x) — Mínimo real vs Nelder-Mead')
plt.legend(); plt.grid(True, linestyle='--', alpha=0.7); plt.tight_layout()
savefig('q1_nelder_mead.png')

# ── 1B: Sensibilidade ao ponto inicial ────────────────────────────────────────
pontos_iniciais = [0, 2, 4, 6, 8, 9, 9.5]
dados_tabela    = []

for x0 in pontos_iniciais:
    res_nm = minimize(f_q1, x0, method='nelder-mead', options={'xatol': 1e-4})
    x_nm, f_nm = res_nm.x[0], res_nm.fun
    status_nm  = "Global" if abs(f_nm - f_real) < 1e-2 else "Local"

    low, high  = max(0, x0 - 1), min(10, x0 + 1)
    res_bt     = minimize_scalar(f_q1, bounds=(low, high), method='bounded')
    x_bt, f_bt = res_bt.x, res_bt.fun
    status_bt  = "Global" if abs(f_bt - f_real) < 1e-2 else "Local"

    dados_tabela.append([x0,
                         f"{x_nm:.3f}", f"{f_nm:.3f}", status_nm,
                         f"{x_bt:.3f}", f"{f_bt:.3f}", status_bt])

fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
colunas = ['Chute (x0)',
           'NM: x*', 'NM: f(x*)', 'NM Status',
           'Brent: x*', 'Brent: f(x*)', 'Brent Status']
tab = ax.table(cellText=dados_tabela, colLabels=colunas,
               loc='center', cellLoc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(10)
tab.scale(1.2, 1.8)
for (row, col), cell in tab.get_celld().items():
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4c72b0')
plt.title(f'Comparativo de Convergência — f(x)=x·sin(5x)  (Mín. real: {f_real:.3f})',
          pad=20, weight='bold')
plt.tight_layout()
savefig('q1b_sensibilidade.png')

# ══════════════════════════════════════════════════════════════════════════════
# QUESTÃO 2
# ══════════════════════════════════════════════════════════════════════════════
dados_q2 = [
    {'x1':  1, 'x2':  1, 'y': 43.614},
    {'x1':  2, 'x2':  4, 'y': 563.694},
    {'x1': -1, 'x2':  2, 'y': 43.230},
    {'x1':  2, 'x2': -2, 'y': 23.130},
]

def model(x1, x2, theta):
    t1, t2, t3, t4 = theta
    return (t1*x1 + t2*np.exp(-(x2**2)) +
            t3*np.log(1 + np.abs(x2)) + t4*(x1**x2))

def g_q2(theta):
    return sum((model(p['x1'], p['x2'], theta) - p['y'])**2 for p in dados_q2)

history_theta, history_g = [], []

def callback(theta):
    history_theta.append(theta.copy())
    history_g.append(g_q2(theta))

res_q2 = minimize(g_q2, [1, 1, 1, 1], method='nelder-mead', callback=callback,
                  options={'xatol': 1e-10, 'fatol': 1e-11,
                           'maxiter': 100000, 'disp': True})

iters      = np.arange(1, len(history_g) + 1)
thetas_arr = np.array(history_theta)

print("=" * 45)
print(f"Iterações necessárias : {len(history_g)}")
print(f"g(θ̂) final            : {res_q2.fun:.8f}")
for i, v in enumerate(res_q2.x, 1):
    print(f"  θ_{i} = {v:.6f}")

# g(θ) por iteração
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(iters, history_g, color='steelblue', linewidth=1)
ax.set_xlabel('Iteração'); ax.set_ylabel('g(θ)')
ax.set_title('Convergência de g(θ) ao longo das iterações')
ax.set_yscale('log'); ax.grid(True); plt.tight_layout()
savefig('q2_convergencia_g.png')

# θ_i por iteração
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes   = axes.flatten()
colors = ['steelblue', 'tomato', 'seagreen', 'darkorange']
for j in range(4):
    axes[j].plot(iters, thetas_arr[:, j], color=colors[j], linewidth=1)
    axes[j].axhline(res_q2.x[j], color='black', linestyle='--', linewidth=1,
                    label=f'θ̂_{j+1} = {res_q2.x[j]:.4f}')
    axes[j].set_xlabel('Iteração'); axes[j].set_ylabel(f'θ_{j+1}')
    axes[j].set_title(f'Trajetória de θ_{j+1}')
    axes[j].legend(fontsize=9); axes[j].grid(True)
plt.suptitle('Evolução dos parâmetros θ', fontsize=12)
plt.tight_layout()
savefig('q2_trajetoria_thetas.png')


fig, ax = plt.subplots(figsize=(5, 3))
ax.axis('off')

cell_text = [
    [f'Iterações',  len(history_g)],
    ['g(θ̂) final', (res_q2.fun)],
    [f'θ₁',         res_q2.x[0]],
    [f'θ₂',         res_q2.x[1]],
    [f'θ₃',         res_q2.x[2]],
    [f'θ₄',         res_q2.x[3]],
]

tab = ax.table(cellText=cell_text, colLabels=['Parâmetro', 'Valor'],
               loc='center', cellLoc='center')
tab.auto_set_font_size(False)
tab.set_fontsize(11)
tab.scale(1.2, 1.8)

for col in range(2):
    tab[0, col].set_facecolor('#2C2C2A')
    tab[0, col].set_text_props(color='white', fontweight='bold')

for row in range(1, len(cell_text) + 1):
    bg = '#f7f6f2' if row % 2 == 0 else 'white'
    for col in range(2):
        tab[row, col].set_facecolor(bg)

plt.title('Resultado — Nelder-Mead', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'q2_resultado_final.png'), dpi=150, bbox_inches='tight')
plt.show()

plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# QUESTÃO 3
# ══════════════════════════════════════════════════════════════════════════════
N_NODES   = 30
N_SAMPLES = 1_000_000
TEORICO   = 1 / np.sqrt(np.pi)

np.random.seed(42)
x_shared = np.random.standard_normal(N_SAMPLES)
y_shared = np.random.standard_normal(N_SAMPLES)

def gauss_hermite_max(n_nodes=N_NODES):
    nodes, weights = hermgauss(n_nodes)
    nodes_norm     = nodes * np.sqrt(2)
    weights_norm   = weights / np.sqrt(np.pi)
    x_g, y_g      = np.meshgrid(nodes_norm, nodes_norm)
    w_x, w_y      = np.meshgrid(weights_norm, weights_norm)
    return np.sum(np.maximum(x_g, y_g) * w_x * w_y)

def monte_carlo_max(x, y):
    u = np.maximum(x, y)
    return np.mean(u), np.std(u) / np.sqrt(len(u))

start = time.time(); resultado_gh = gauss_hermite_max(); tempo_gh = time.time() - start
start = time.time(); resultado_mc, se_mc = monte_carlo_max(x_shared, y_shared); tempo_mc = time.time() - start

dados_gh = [["Resultado",     f"{resultado_gh:.8f}"],
            ["Valor Teórico", f"{TEORICO:.8f}"],
            ["Erro Absoluto", f"{abs(resultado_gh - TEORICO):.2e}"],
            ["Tempo (s)",     f"{tempo_gh:.6f}"]]
dados_mc = [["Resultado",        f"{resultado_mc:.8f}"],
            ["Valor Teórico",    f"{TEORICO:.8f}"],
            ["Erro Real",        f"{abs(resultado_mc - TEORICO):.2e}"],
            ["Erro Padrão (SE)", f"{se_mc:.2e}"],
            ["Tempo (s)",        f"{tempo_mc:.6f}"]]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
for ax, dados, title, cor in [
    (ax1, dados_gh, f"Gauss-Hermite ({N_NODES} nós)",       '#d1e7dd'),
    (ax2, dados_mc, f"Monte Carlo ({N_SAMPLES:,} amostras)", '#fff3cd'),
]:
    ax.axis('off')
    ax.set_title(title, weight='bold', pad=10)
    tab = ax.table(cellText=dados, colLabels=["Métrica", "Valor"],
                   loc='center', cellLoc='left')
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.6)
    for (row, col), cell in tab.get_celld().items():
        if row == 0: cell.set_facecolor(cor)
plt.suptitle("E[max(X,Y)]  com  X,Y ~ N(0,1)", fontsize=12, weight='bold')
plt.tight_layout()
savefig('q3_tabela_integracao.png')

# ══════════════════════════════════════════════════════════════════════════════
# QUESTÃO 4
# ══════════════════════════════════════════════════════════════════════════════
def trapezio(f, a, b, n):
    x = np.linspace(a, b, n + 1); y = f(x)
    h = (b - a) / n
    return h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2), x, y

def plot_trapezio(f, a, b, n, filename):
    integral, x_trap, y_trap = trapezio(f, a, b, n)
    x_smooth = np.linspace(a, b, 1000)
    fig, ax  = plt.subplots(figsize=(10, 5))
    ax.fill_between(x_trap, y_trap, alpha=0.3, color='steelblue', label='Área aproximada')
    for i in range(len(x_trap)):
        ax.plot([x_trap[i], x_trap[i]], [0, y_trap[i]],
                color='steelblue', linewidth=0.7, alpha=0.5)
    for i in range(len(x_trap) - 1):
        ax.plot([x_trap[i], x_trap[i+1]], [y_trap[i], y_trap[i+1]],
                color='steelblue', linewidth=1.2)
    ax.plot(x_smooth, f(x_smooth), color='tomato', linewidth=2, label='f(x)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel('x'); ax.set_ylabel('f(x)')
    ax.set_title(f'Trapézio  |  n={n}  |  Integral ≈ {integral:.6f}')
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    savefig(filename)
    return integral

n_breaks = [3, 5, 10, 15, 20]

def fa(x): return x
def intf(x1, x2): return x2**2/2 - x1**2/2

def gb(x): return x * np.sin(x)
def intg(x1, x2): return (np.sin(x2) - x2*np.cos(x2)) - (np.sin(x1) - x1*np.cos(x1))

def hc(x): return np.sqrt(1 - x**2)

resultado_a = plot_trapezio(fa, 0, 1, n=2,   filename='q4a_trapezio.png')
resultado_b = plot_trapezio(gb, 0, 1, n=5,   filename='q4b_trapezio.png')
resultado_c = plot_trapezio(hc, 0, 1, n=100, filename='q4c_trapezio.png')

result_analitico_a = intf(0, 1)
result_analitico_b = intg(0, 1)
result_analitico_c = np.pi / 4

erros_funcs = [
    ([abs(trapezio(fa, 0, 1, n)[0] - result_analitico_a) for n in n_breaks], 'f(x) = x',        '#d1e7dd', '#0a3622'),
    ([abs(trapezio(gb, 0, 1, n)[0] - result_analitico_b) for n in n_breaks], 'g(x) = x·sin(x)', '#fff3cd', '#664d03'),
    ([abs(trapezio(hc, 0, 1, n)[0] - result_analitico_c) for n in n_breaks], 'h(x) = √(1-x²)',  '#cfe2ff', '#084298'),
]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (erros, label, bg, hc_cor) in zip(axes, erros_funcs):
    ax.axis('off'); ax.set_title(label, fontsize=11, fontweight='bold', pad=12)
    tab = ax.table(cellText=[[str(n), f"{e:.2e}"] for n, e in zip(n_breaks, erros)],
                   colLabels=['n', 'Erro absoluto'], loc='center', cellLoc='center')
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 1.8)
    for col in range(2):
        tab[0, col].set_facecolor(hc_cor)
        tab[0, col].set_text_props(color='white', fontweight='bold')
    for row in range(1, len(n_breaks) + 1):
        for col in range(2):
            tab[row, col].set_facecolor(bg if row % 2 == 0 else 'white')
plt.suptitle('Erro absoluto do Trapézio por n', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
savefig('q4_tabela_erros.png')

# ══════════════════════════════════════════════════════════════════════════════
# QUESTÃO 5
# ══════════════════════════════════════════════════════════════════════════════
def diff_centrada_2pts(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2*h)

def diff_centrada_4pts(f, x, h=1e-5):
    return (-f(x + 2*h) + 8*f(x + h) - 8*f(x - h) + f(x - 2*h)) / (12*h)

h_list = [0.001, 0.005, 0.01, 0.05]

def f1(x): return x**2
def derivf1(x): return 2*x

def f2(x): return np.log(x)
def derivf2(x): return 1/x

def f3(x): return x * np.sin(x)
def derivf3(x): return np.sin(x) + x*np.cos(x)

funcs5  = [(f1, derivf1, 5), (f2, derivf2, 10), (f3, derivf3, 1)]
labels5 = ['f₁(x)=x²  x=5', 'f₂(x)=ln(x)  x=10', 'f₃(x)=x·sin(x)  x=1']

results5 = []
for f, df, x0 in funcs5:
    analitico = df(x0)
    rows = [(h, diff_centrada_2pts(f, x0, h), diff_centrada_4pts(f, x0, h))
            for h in h_list]
    results5.append((analitico, rows))

HEADER_BG = '#2C2C2A'; HEADER_FG = '#F1EFE8'
ANALI_BG  = '#cfe2ff'; ANALI_FG  = '#084298'
ROW_ALT   = '#f7f6f2'; ROW_WHITE = '#ffffff'

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (analitico, rows), label in zip(axes, results5, labels5):
    ax.axis('off'); ax.set_title(label, fontsize=10, fontweight='bold', pad=10)
    cell_text = [['analítico', f'{analitico:.8f}', '—', f'{analitico:.8f}', '—']]
    for h, d2p, d4p in rows:
        cell_text.append([str(h), f'{d2p:.8f}', f'{abs(analitico-d2p):.2e}',
                                  f'{d4p:.8f}', f'{abs(analitico-d4p):.2e}'])
    tab = ax.table(cellText=cell_text,
                   colLabels=['h', '2 pontos', 'erro 2pts', '4 pontos', 'erro 4pts'],
                   loc='center', cellLoc='center')
    tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.8)
    for col in range(5):
        tab[0, col].set_facecolor(HEADER_BG)
        tab[0, col].set_text_props(color=HEADER_FG, fontweight='bold')
        tab[1, col].set_facecolor(ANALI_BG)
        tab[1, col].set_text_props(color=ANALI_FG, fontweight='bold')
    for row in range(2, len(cell_text) + 1):
        bg = ROW_ALT if row % 2 == 0 else ROW_WHITE
        for col in range(5):
            tab[row, col].set_facecolor(bg)
plt.suptitle('Diferenças Finitas Centradas — 2 e 4 pontos', fontsize=12,
             fontweight='bold', y=1.02)
plt.tight_layout()
savefig('q5_tabela_derivadas.png')

print("\n✔  Todos os plots salvos em:", out_dir)