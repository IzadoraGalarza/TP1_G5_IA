# Parte 1 — Redundantes, inconsistentes e estrutura comum de experimentos (Izadora)

**Pergunta:** como identificar e tratar dados redundantes, inconsistentes ou suspeitos de vazamento antes de treinar os modelos?

**Resposta curta:** realizar uma limpeza progressiva dos dados, identificando duplicatas, conflitos, atributos redundantes, valores fora dos limites esperados e possíveis variáveis com vazamento. A estratégia `F_E_mais_suspeitas` reúne os tratamentos mais completos.

## O que foi analisado

A auditoria dos dados verificou:

| **Verificação** | **O que procura** |
|---|---|
| Redundâncias matemáticas | Colunas derivadas de outras, como `total_bytes` |
| Colunas constantes | Atributos sem variação |
| Duplicatas | Registros repetidos |
| Rótulos conflitantes | Mesmos atributos associados a classes diferentes |
| Valores fora do domínio | Contagens e taxas fora dos limites esperados |
| Dados ausentes | Valores que precisam de tratamento |
| Possível vazamento | Variáveis que podem conter informações indisponíveis no momento da previsão |

### Principais achados

- Foram identificados **1.702 grupos com atributos coincidentes e classes divergentes**, envolvendo 3.730 linhas.
- `src_bytes` apresenta aproximadamente **39,96% de valores ausentes**.
- `tempo_resposta_analista_min` apresenta aproximadamente **91,37% de valores ausentes**.
- Foram identificados atributos matematicamente derivados, como `src_bytes_kb`, `dst_bytes_kb` e `total_bytes`.
- Foram levantadas suspeitas sobre `tipo_alerta_soc`, `score_ids_legado` e `tempo_resposta_analista_min`, que precisam ter sua disponibilidade no momento da previsão confirmada.

## Estratégias de limpeza comparadas

| **Estratégia** | **O que faz** |
|---|---|
| `A_sem_limpeza` | Mantém os dados sem as correções propostas |
| `B_normaliza_faixa` | Padroniza categorias e trata valores fora dos limites como ausentes |
| `C_B_mais_dedup` | Adiciona a remoção de duplicatas exatas |
| `D_C_sem_conflitos` | Remove registros pertencentes a grupos com rótulos conflitantes |
| `E_D_mais_redundantes` | Remove atributos matematicamente redundantes |
| `F_E_mais_suspeitas` | Também remove atributos suspeitos de vazamento |

As estratégias foram avaliadas utilizando Random Forest, mantendo fixos os demais parâmetros: codificação `onehot_rare`, normalização Standard, sem balanceamento e validação cruzada com três dobras.

## O que isso significa

1. **A limpeza deve ser progressiva.** Comparar as seis estratégias permite observar como cada tratamento interfere no desempenho do modelo.
2. **Duplicatas e conflitos precisam de atenção.** Registros repetidos podem influenciar a avaliação, enquanto registros com os mesmos atributos e rótulos diferentes indicam inconsistências.
3. **Atributos redundantes podem ser removidos.** Colunas matematicamente derivadas podem acrescentar complexidade sem trazer informação nova.
4. **Possíveis vazamentos precisam ser investigados.** Variáveis que só existem depois da identificação de uma anomalia não devem ser utilizadas para prever essa mesma anomalia.
5. **O tratamento deve evitar data leakage.** Imputação, codificação e normalização devem ser ajustadas dentro do pipeline, utilizando somente os dados de treino de cada dobra.

**Resultado gerado:** `resultados_parte1_limpeza.csv`.

**Observação:** a tabela de resultados da execução deve ser consultada para identificar qual estratégia apresentou o melhor F1-macro. Não é possível afirmar uma vencedora sem esses valores.

## Como isso afeta as outras partes

| **Parte** | **Relação** |
|---|---|
| 2 · Codificação | A padronização das categorias evita categorias duplicadas por diferenças de espaços e capitalização. |
| 3 · Normalização | A base preparada é utilizada para comparar Standard, MinMax, Log e Robust. |
| 4 · Balanceamento | A base limpa serve como entrada para comparar as técnicas de balanceamento. |
| 5 · Redução | A remoção de redundâncias ajuda a preparar os dados para PCA ou seleção de instâncias. |

### Ponto para revisão

A remoção de duplicatas e conflitos antes da validação cruzada pode deixar a avaliação otimista, pois esses registros deixam de participar das dobras. Essa limitação deve ser considerada na interpretação dos resultados.

---
## Resumo geral

- **Parte 1 — Limpeza:** identificar e tratar problemas na base, comparando seis estratégias de limpeza.
