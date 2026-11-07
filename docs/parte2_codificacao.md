# Parte 2 — Codificação categórica (Samuel Alves)

**Pergunta:** como transformar `protocol_type`, `service` e `flag` em números sem perder informação e
sem vazamento?

**Resposta curta:** com Random Forest (modelo da equipe), use **`encoder='ordinal'`**. Com modelos
lineares ou baseados em distância (LR, KNN, redes), use **`encoder='onehot'`**, o padrão da estrutura
comum, que já agrupa as categorias raras.

Notebook: [`notebooks/parte2_codificacao.ipynb`](../notebooks/parte2_codificacao.ipynb) ·
[abrir no Colab](https://colab.research.google.com/github/IzadoraGalarza/TP1_G5_IA/blob/main/notebooks/parte2_codificacao.ipynb)

## O que foi comparado

Mantendo fixos a limpeza (`F_E+leakage`), a escala (`standard`), o balanceamento (`nenhum`) e a validação
da estrutura comum, mudando só a codificação:

| Variante | Como funciona | Colunas geradas |
|---|---|---|
| `onehot_puro` | uma coluna por categoria | 79 |
| `onehot` | one-hot com categorias de menos de 20 linhas agrupadas em "infrequente" | 36 |
| `ordinal` | um inteiro por categoria (ordem alfabética) | 3 |
| `frequency` | frequência da categoria no treino do fold | 3 |
| `onehot` + `ausente` | valor faltante vira a categoria `ausente` em vez da moda | 36 |

Cada variante rodou com **RF** e com **LR**.

## Resultados (F1-macro, 5 folds)

| Codificação | RF | LR |
|---|---|---|
| `onehot` | **0,9765** ± 0,0035 | 0,9317 ± 0,0036 |
| `onehot` + `ausente` | 0,9763 ± 0,0029 | **0,9340** ± 0,0093 |
| `ordinal` | 0,9753 ± 0,0047 | 0,9075 ± 0,0040 |
| `frequency` | 0,9749 ± 0,0035 | 0,9006 ± 0,0045 |
| `onehot_puro` | 0,9740 ± 0,0042 | 0,9309 ± 0,0041 |

Tabela completa: [`output/etapa_2/resultados_parte2_codificacao.csv`](../output/etapa_2/resultados_parte2_codificacao.csv)

## O que isso significa

1. **No RF, as codificações empatam.** Todas as diferenças são menores que um desvio-padrão. Árvores
   separam categorias por limiares, então a ordem arbitrária do ordinal quase não atrapalha. Pela regra
   definida antes de ver os resultados (entre as empatadas, a de menos colunas), fica **ordinal**:
   3 colunas em vez de 36.
2. **Na LR, o one-hot é claramente melhor** (+0,025 de F1-macro). Um modelo linear dá um único peso ao
   código numérico e não consegue isolar serviços raros e suspeitos: `private` tem 34,5% de anomalias,
   `ecr_i` 32,7% e `eco_i` 25,2%, contra 0,4% em `http`.
3. **Agrupar as categorias raras não custa nada.** O one-hot cai de 79 para 36 colunas sem perder desempenho.
4. **Imputação pela moda ou com a categoria `ausente` dá no mesmo.** A taxa de ausência é quase igual nas
   duas classes (6,8% × 6,3% em `service`), então faltar não indica a classe.
5. **Sem vazamento.** Nenhuma técnica usa o rótulo (por isso não usamos *target encoding*); vocabulário,
   frequências e moda são aprendidos só no treino de cada fold.

**Achado de dados:** `service` tinha nomes com espaço no final (`'http '`), o que criava 29 categorias
falsas (96 em vez de 67). O `strip` da limpeza da Parte 1 resolve isso, mas vale citar no relatório.

## Como isso afeta as outras partes

| Parte | Recomendação |
|---|---|
| 3 · Normalização | Com LR, use `encoder='onehot'`; com ordinal, a LR perde desempenho por causa da codificação, não da escala. |
| 4 · Balanceamento | `SMOTENC` exige `encoder='ordinal'`, que é justamente a escolha para RF. Para `SMOTE`, use `onehot`. |
| 5 · PCA | Prefira `onehot`: PCA sobre códigos ordinais mistura números sem significado. |

## Como colocar no Colab do grupo

1. Abra o notebook pelo link do Colab acima.
2. Copie as células da **Seção B** (do título "Seção B" até o fim).
3. No Colab do grupo, apague as duas células de exemplo da Parte 2 e cole logo depois da célula que cria
   `train_clean`.
4. Execute. Leva cerca de 4 minutos e gera `resultados_parte2_codificacao.csv` e `parte2_dimensionalidade.png`.

## Pontos para revisar na estrutura comum (Parte 1)

- `clean_train` remove duplicatas e rótulos conflitantes **antes** da validação cruzada, então as dobras de
  validação ficam sem esses casos difíceis. Os F1 de cerca de 0,97 provavelmente estão otimistas em relação
  ao teste.
- `CV` usa `shuffle=True`, e o enunciado diz que não é permitido re-embaralhar.
