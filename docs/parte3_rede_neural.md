# Etapa 3 - Rede Neural

Responsável: Samuel Ferreira Alves Vieira.

## Objetivo

Avaliar o `MLPClassifier` variando arquitetura, regularização L2 (`alpha`) e uso de
`early_stopping`, mantendo fixo o pré-processamento escolhido na Etapa 2.

## Protocolo

- base oficial `data/treino.csv`;
- limpeza `F_E_mais_suspeitas`: normalização categórica, valores inválidos convertidos em
  ausentes, remoção de duplicatas, conflitos, redundâncias e três atributos com risco de
  vazamento;
- 55.063 linhas e 39 preditores após a limpeza;
- imputação por moda/mediana dentro de cada dobra;
- `onehot_rare` para categóricas;
- `StandardScaler` para numéricas, conforme a normalização escolhida na Etapa 2;
- `StratifiedGroupKFold` com 3 dobras, embaralhamento e semente 42;
- assinaturas idênticas mantidas na mesma dobra;
- sem uso do conjunto de teste e sem balanceamento nesta comparação.

Foram avaliadas 18 combinações:

- arquiteturas: `(64,)`, `(64, 32)` e `(128,)`;
- `alpha`: `0.0001`, `0.001` e `0.01`;
- `early_stopping`: ligado e desligado.

Os demais parâmetros foram fixados: ReLU, Adam, lote 256, taxa inicial 0,001, máximo de 180
iterações, tolerância `1e-4` e semente 42.

## Melhor configuração

| Parâmetro | Valor |
|---|---:|
| Camadas ocultas | `(128,)` |
| `alpha` | 0,0001 |
| `early_stopping` | não |
| F1-macro | 0,9656 ± 0,0052 |
| Precisão - normal | 0,9970 |
| Recall - normal | 0,9986 |
| Precisão - anomaly | 0,9571 |
| Recall - anomaly | 0,9108 |
| Iterações médias | 94 |

A configuração `(128,)`, `alpha=0.001`, também sem early stopping, ficou próxima: F1-macro
0,9654 e recall de anomalia 0,9092. A melhor rede anterior, `(64,)` com `alpha=0.001`, obteve
F1-macro 0,9638 e recall de anomalia 0,8988.

## Discussão

Desligar o early stopping produziu os maiores F1-macro nas três arquiteturas. Quando ligado, o
treinamento terminou em aproximadamente 25 a 36 iterações, contra 81 a 116 sem ele, porém o recall
da classe `anomaly` caiu para a faixa de 0,8530 a 0,8781. Portanto, o ganho de tempo não compensou a
perda na detecção da classe minoritária neste experimento.

A segunda camada não trouxe ganho consistente. A rede `(128,)` aumentou modestamente o F1-macro
e o recall de anomalia em relação à rede `(64,)`, com pequena perda de precisão na classe
minoritária. A diferença entre os dois melhores `alpha` é menor que o desvio entre as dobras; por
isso, a conclusão correta é que ambos foram estáveis, não que 0,0001 seja estatisticamente
superior. Para a tabela comparativa da equipe, levamos `(128,)`, `alpha=0.0001` e
`early_stopping=False` por obter a maior média observada e melhor recall de anomalia.

## Arquivos

- `notebooks/parte3_rede_neural.ipynb`: notebook reproduzível;
- `output/etapa_3/resultados_rede_neural.csv`: uma linha por combinação;
- `output/etapa_3/resultados_rede_neural_por_dobra.csv`: métricas de cada dobra;
- `output/etapa_3/rede_neural_f1_macro.png`: comparação de F1-macro;
- `output/etapa_3/rede_neural_precisao_recall_anomaly.png`: compromisso entre precisão e recall;
- `src/stage3_neural_network.py`: pipeline e execução por linha de comando.

## Texto curto para a tabela/discussão da equipe

Foram testadas 18 configurações de MLP com StandardScaler, variando três arquiteturas, três valores
de `alpha` e early stopping. A melhor média foi obtida por `(128,)`, `alpha=0.0001` e
`early_stopping=False`: F1-macro 0,9656 ± 0,0052, precisão/recall de normal 0,9970/0,9986 e de
anomaly 0,9571/0,9108. O early stopping reduziu o tempo e as iterações, mas piorou principalmente o
recall de anomalias. A arquitetura com duas camadas não melhorou consistentemente o F1-macro.
