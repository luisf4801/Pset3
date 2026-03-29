# PSET 3 — Métodos Numéricos

Implementação das questões 1–5 do Problem Set 3, cobrindo otimização, integração numérica e diferenciação numérica.

---

## Estrutura de Pastas

```
pset3/
├── pset3.py            ← código principal (todas as questões)
├── pset3_comp.PDF         ← PDF com resultados
├── requirements.txt    ← dependências do projeto
├── README.md           ← este arquivo
└── output/             ← gerada automaticamente ao rodar o script
    ├── q1_grid_search.png
    ├── q1_brent.png
    ├── q1_nelder_mead.png
    ├── q1b_sensibilidade.png
    ├── q2_convergencia_g.png
    ├── q2_trajetoria_thetas.png
    ├── q2_resultado_final.png
    ├── q3_tabela_integracao.png
    ├── q4a_trapezio.png
    ├── q4b_trapezio.png
    ├── q4c_trapezio.png
    ├── q4_tabela_erros.png
    └── q5_tabela_derivadas.png
```

A pasta `output/` não precisa ser criada manualmente — o script cria automaticamente ao rodar.

---

## Como Recriar o Ambiente

### Usando Conda (recomendado)

```bash
# 1. Criar o ambiente
conda create -n pset3 python=3.11

# 2. Ativar
conda activate pset3

# 3. Instalar dependências
pip install -r requirements.txt
```

### Usando venv

```bash
# 1. Criar o ambiente
python -m venv venv

# 2. Ativar (Windows)
venv\Scripts\activate

# 3. Instalar dependências
pip install -r requirements.txt
```

### Rodar o script

```bash
python pset3.py
```

Os gráficos serão salvos automaticamente em `output/`.

---

## Dependências

| Pacote | Uso |
|--------|-----|
| `numpy` | Operações numéricas, arrays, geração de nós Gauss-Hermite |
| `matplotlib` | Geração de todos os gráficos e tabelas |
| `scipy` | Métodos de otimização (Brent, Nelder-Mead) |
| `time` | Medição de tempo de execução (biblioteca padrão) |
| `os` | Manipulação de caminhos e criação de pastas (biblioteca padrão) |

---

## O que cada Questão faz

### Questão 1 — Minimização de f(x) = x·sin(5x)

Compara três métodos de otimização escalar no intervalo [0, 10]:

- **Grid Search** — avalia a função em 100.000 pontos uniformemente espaçados e retorna o mínimo. Simples mas custoso e limitado pela resolução da grade.
- **Brent** — método de busca por intervalo usando `scipy.optimize.minimize_scalar`. Combina busca pela razão áurea e interpolação parabólica. Convergência garantida, sem necessidade de derivada.
- **Nelder-Mead** — método simplex que não usa derivadas. Sensível ao ponto inicial, podendo convergir para mínimos locais.

A parte **1B** varia o ponto inicial de Nelder-Mead e Brent em sete valores e registra em tabela se cada método encontrou o mínimo global ou ficou preso num local.

Gráficos gerados: `q1_grid_search.png`, `q1_brent.png`, `q1_nelder_mead.png`, `q1b_sensibilidade.png`

---

### Questão 2 — Estimação de parâmetros por Nelder-Mead

Dado o modelo não-linear:

```
y = θ₁x₁ + θ₂·exp(-x₂²) + θ₃·ln(1 + |x₂|) + θ₄·x₁^x₂
```

Minimiza a soma dos quadrados dos resíduos `g(θ) = Σ(ŷ - y)²` sobre 4 observações usando Nelder-Mead. Um callback registra o valor de `g(θ)` e os parâmetros a cada iteração, permitindo visualizar a trajetória de convergência.

Gráficos gerados: `q2_convergencia_g.png`, `q2_trajetoria_thetas.png`, `q2_resultado_final.png`

---

### Questão 3 — Integração: E[max(X, Y)] com X, Y ~ N(0,1)

Calcula o valor esperado de max(X, Y) onde X e Y são normais padrão independentes. O valor analítico é `1/√π ≈ 0.5642`.

Dois métodos são comparados:

- **Gauss-Hermite** — quadratura determinística com 30 nós. Usa `numpy.polynomial.hermite.hermgauss`. Alta precisão com custo computacional baixo.
- **Monte Carlo** — média amostral sobre 1.000.000 de realizações. Inclui erro padrão estimado. Mais lento, convergência estocástica.

A tabela compara resultado, erro absoluto e tempo de execução de cada método.

Gráfico gerado: `q3_tabela_integracao.png`

---

### Questão 4 — Integração numérica pela Regra do Trapézio

Implementa a regra do trapézio composta e aplica a três funções:

- **f(x) = x** em [0, 1] — função linear, o trapézio é exato
- **g(x) = x·sin(x)** em [0, 1] — função suave, boa convergência
- **h(x) = √(1-x²)** em [0, 1] — quarto de círculo, resultado teórico = π/4

Para cada função, gera um gráfico ilustrando a aproximação trapezoidal e uma tabela de erro absoluto para `n ∈ {3, 5, 10, 15, 20}` subintervalos.

Gráficos gerados: `q4a_trapezio.png`, `q4b_trapezio.png`, `q4c_trapezio.png`, `q4_tabela_erros.png`

---

### Questão 5 — Diferenciação numérica por diferenças finitas centradas

Compara dois esquemas de diferenças finitas centradas para três funções em pontos específicos:

- **2 pontos**: `f'(x) ≈ (f(x+h) - f(x-h)) / 2h` — erro de ordem O(h²)
- **4 pontos**: `f'(x) ≈ (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / 12h` — erro de ordem O(h⁴)

Funções avaliadas:

| Função | Ponto | Derivada analítica |
|--------|-------|-------------------|
| f₁(x) = x² | x = 5 | 2x = 10 |
| f₂(x) = ln(x) | x = 10 | 1/x = 0.1 |
| f₃(x) = x·sin(x) | x = 1 | sin(x) + x·cos(x) |

Para cada função, uma tabela mostra o valor aproximado e o erro absoluto em relação à derivada analítica para `h ∈ {0.001, 0.005, 0.01, 0.05}`.

Gráfico gerado: `q5_tabela_derivadas.png`
