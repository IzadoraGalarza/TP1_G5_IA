# Status da Etapa 2 — o que já está feito e o que falta

Atualizado em 07/11/2026. A divisão completa (responsabilidades, entregáveis e revisões) está em
[`etapa2_divisao.md`](etapa2_divisao.md). Este arquivo é o ponto de partida para quem vai começar a
sua parte.

## Resumo rápido

| Parte | Responsável | Situação | Onde está |
|---|---|---|---|
| 1. Redundâncias e inconsistências | Daniel Alves | 🟡 rascunho pronto e executado; falta assumir, revisar e escrever a justificativa | `src/cleaning.py`, seção de limpeza do `Trabalho_IA_Etapa_2.ipynb` |
| 2. Ausentes e codificação categórica | Samuel | ✅ concluída (falta só a revisão cruzada) | `src/encoding.py`, `notebooks/02_codificacao.ipynb` |
| 3. Transformação numérica | Daniel Pedrosa | ⚪ não iniciada | — |
| 4. Balanceamento | Izadora | 🟡 rascunho (sem × RandomUnder × RandomOver) executado; falta análise, gráfico e texto | `evaluate_preprocessing_strategies` em `src/preprocessing_stage2.py` |
| 5. Redução de dimensionalidade | Larissa | 🟡 SVD com 20 componentes fixo no rascunho; falta comparar 10/20/40 e a variância | `build_experiment_pipeline` em `src/preprocessing_stage2.py` |

## Base comum (não precisa refazer)

- **Dados**: os três CSVs do Canvas já estão no repositório, em `data/`; basta clonar.
  `python -m src.data_validation --data-dir data` confere o contrato (deve imprimir “Contrato dos dados válido”).
- **Preparação determinística**: `prepare_modeling_data(df)` em `src/preprocessing_stage2.py` remove as
  duplicatas exatas, as colunas suspeitas de vazamento (`tipo_alerta_soc`, `score_ids_legado`,
  `tempo_resposta_analista_min`), a constante e as derivadas de bytes, normaliza as categorias
  (`strip` + `lower`) e troca valores inválidos por `NaN`. Devolve `X`, `y` e um relatório.
- **Controle**: `build_preprocessor(report)`, com moda + one-hot nas categóricas e mediana + indicador +
  `StandardScaler` nos numéricos.
- **Validação**: `StratifiedGroupKFold(n_splits=5, shuffle=False)` com `groups=make_duplicate_groups(X)`.
  Use sempre essa configuração, para a tabela final ser comparável.
- **Métricas**: `SCORING` em `src/experiments.py` (F1-macro, precisão/recall macro, F1, precisão e recall de `anomaly`).
- **Classificador de referência**: `LogisticRegression(max_iter=2000, random_state=42)`.

Modelo a seguir: `src/encoding.py` + `notebooks/02_codificacao.ipynb`. Ali estão as variantes declaradas
numa lista, `cross_validate` com `groups`, a tabela salva em `output/etapa_2/resultados_<parte>.csv`, a
figura, uma regra de decisão fixada antes de ver os resultados e um teste em `tests/`.

## Resultados já obtidos (base oficial, 69.833 linhas)

**Parte 2 — codificação** (`output/etapa_2/resultados_codificacao.csv`)

| Variante | F1-macro | Recall anomaly | Colunas categóricas |
|---|---|---|---|
| `ausente` + one-hot | 0,8037 ± 0,0046 | 0,555 | 83 |
| **one-hot com raras agrupadas (escolhida)** | 0,8018 ± 0,0042 | 0,550 | 30 |
| controle: moda + one-hot | 0,8015 ± 0,0050 | 0,550 | 81 |
| híbrida (frequência só em `service`) | 0,7941 ± 0,0030 | 0,527 | 15 |
| ordinal | 0,7940 ± 0,0029 | 0,527 | 3 |
| frequência | 0,7936 ± 0,0021 | 0,526 | 3 |

**Rascunho da Parte 1 — limpeza** (`output/etapa_2/resultados_limpeza.csv`): remover os rótulos
conflitantes do treino deu o maior F1-macro (0,8745 contra 0,8370 sem tratamento); as outras variantes
ficaram entre 0,827 e 0,837. ⚠️ Confirmar na revisão que a remoção acontece só no treino de cada dobra.

**Rascunho da Parte 4 — balanceamento** (`output/etapa_2/resultados_preprocessamento.csv`, com SVD-20 fixo):

| Estratégia | F1-macro | Recall anomaly |
|---|---|---|
| sem balanceamento | 0,7814 ± 0,0075 | 0,487 |
| RandomOverSampler | 0,6844 ± 0,0061 | 0,727 |
| RandomUnderSampler | 0,6820 ± 0,0075 | 0,731 |

## Achados que valem para todos

- `service` tinha nomes com espaço no final (`'http '`), o que criava 29 categorias falsas (96 em vez de 67).
  A EDA da Etapa 1 não detectou isso; o `strip` da preparação comum resolve.
- `service` e `flag` têm cerca de 6% de valores ausentes, com taxa quase igual nas duas classes.
- Corrigido: categóricas ausentes travavam o `SimpleImputer` (`pd.NA`). Se aparecer
  `boolean value of NA is ambiguous`, atualize a branch.
- Entre as dobras surgem pouquíssimas categorias novas, e o teste não traz nenhuma categoria ausente
  do treino; mesmo assim mantenha `handle_unknown` no codificador.

## O que falta para fechar a entrega

1. **Daniel Pedrosa**: Parte 3 inteira (sem escala × Standard × MinMax × Robust, `log1p` nos assimétricos).
2. **Daniel Alves, Izadora e Larissa**: assumir os rascunhos, completar as variantes pedidas na divisão,
   gerar figura e parágrafo e escrever um teste.
3. **Larissa**: juntar as seções no `Trabalho_IA_Etapa_2.ipynb` e executar “Run all”.
4. **Izadora**: consolidar as tabelas de `output/etapa_2/` numa única tabela comparativa.
5. **Todos**: fazer a revisão cruzada (ver `etapa2_divisao.md`). Samuel revisa a limpeza do Daniel Alves.
6. **Samuel**: conferir no notebook final se todos os requisitos do item 4 do enunciado aparecem.

## Como rodar

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests          # 24 testes devem passar
jupyter lab                                   # abrir os notebooks e executar tudo
```
