

# Parte 3 — Normalização e transformação numérica (Izadora)

**Pergunta:** qual transformação numérica apresenta melhor desempenho para os modelos, considerando que eles podem reagir de maneiras diferentes à escala dos atributos?

**Resposta curta:** para o Random Forest, **Standard** apresentou o melhor F1-macro (0,9742), praticamente empatado com Log. Para a Regressão Logística, **Log** foi a melhor opção, com F1-macro de 0,9379.

## O que foi comparado

Mantendo fixos a limpeza (`F_E+leakage`), a codificação (`onehot_rare`), o balanceamento (`nenhum`) e a validação cruzada de três dobras, foram comparadas quatro transformações:

| **Transformação** | **Como funciona** |
|---|---|
| `standard` | Centraliza os dados pela média e divide pelo desvio-padrão |
| `minmax` | Coloca os valores em uma faixa, geralmente entre 0 e 1 |
| `log` | Aplica transformação logarítmica e depois StandardScaler |
| `robust` | Utiliza mediana e intervalo interquartil, sendo menos sensível a valores extremos |

Cada transformação foi testada com **Regressão Logística (LR)** e **Random Forest (RF)**.

A transformação Log utiliza:

\[
f(x)=sign(x)\times \log(1+|x|)
\]

Essa forma preserva o sinal dos valores e permite aplicar a transformação também em valores negativos.

## Resultados (F1-macro, 3 folds)

| **Normalização** | **RF** | **LR** |
|---|---:|---:|
| `standard` | **0,9742 ± 0,0049** | 0,9322 ± 0,0007 |
| `log` | 0,9741 ± 0,0051 | **0,9379 ± 0,0029** |
| `robust` | 0,9739 ± 0,0054 | 0,9304 ± 0,0019 |
| `minmax` | 0,9735 ± 0,0054 | 0,9287 ± 0,0039 |

Tabela completa: `resultados_parte3_normalizacao.csv`.

## O que isso significa

1. **O Random Forest praticamente não foi afetado pela normalização.** Os resultados ficaram entre 0,9735 e 0,9742, com diferenças muito pequenas. Isso ocorre porque árvores de decisão geralmente são menos sensíveis à escala dos atributos.
2. **A Regressão Logística se beneficiou da transformação Log.** Ela alcançou F1-macro de 0,9379, aproximadamente 0,0057 acima do Standard.
3. **MinMax teve o pior resultado na LR.** Seu F1-macro foi 0,9287, enquanto Robust ficou em 0,9304.
4. **Standard foi a melhor opção para RF.** Apesar disso, sua diferença em relação a Log e Robust foi praticamente insignificante.
5. **O modelo influencia a escolha da transformação.** A normalização teve maior impacto na Regressão Logística do que no Random Forest.

## Como isso afeta as outras partes

| **Parte** | **Relação** |
|---|---|
| 1 · Limpeza | A normalização utiliza a base preparada após a estratégia de limpeza escolhida. |
| 2 · Codificação | A codificação fica fixa em `onehot_rare` para que a comparação avalie principalmente as transformações numéricas. |
| 4 · Balanceamento | A transformação escolhida pode ser mantida fixa para comparar as técnicas de balanceamento. |
| 5 · Redução | A escala dos atributos também deve ser considerada antes de aplicar técnicas como PCA. |

## Conclusão da Parte 3

O Random Forest apresentou o melhor desempenho geral, com F1-macro de 0,9742 utilizando Standard. Já a Regressão Logística teve seu melhor resultado com a transformação Log, alcançando 0,9379.

Conclui-se que a escolha da normalização depende do modelo utilizado. A transformação numérica teve pouca influência sobre o Random Forest, mas apresentou impacto mais perceptível na Regressão Logística.

---

## Resumo geral

- **Parte 3 — Normalização:** investigar como diferentes transformações numéricas influenciam o desempenho dos modelos.


