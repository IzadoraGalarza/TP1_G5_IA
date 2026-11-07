# Etapa 2 - Pré-processamento: divisão entre cinco pessoas

Entrega da etapa (enunciado, item 10): notebook com experimentos de pré-processamento, resultados
comparativos e justificativas. Requisitos mínimos (item 4 do enunciado):

1. tratamento de dados redundantes/inconsistentes;
2. 2 técnicas de balanceamento;
3. 1 técnica de conversão simbólico-numérica, justificada (cardinalidade de `service`);
4. 1 técnica de normalização/transformação numérica;
5. 1 técnica de redução de dimensionalidade **ou** seleção de instâncias.

Cada requisito vira uma parte. A parte 1 também entrega o **harness comum** (`src/experiments.py`)
que as outras usam, para que toda a tabela venha do mesmo protocolo.

## Regras comuns (valem para todas as partes)

- Dados oficiais em `data/` (`treino.csv`, `teste_sem_rotulo.csv`, `modelo_de_submissao.csv`).
  O teste **nunca** é usado para ajustar nada.
- Validação: `make_cv(y)` = `StratifiedKFold(n_splits=5, shuffle=False)`, sempre o mesmo.
- Toda etapa que aprende algo (imputação, codificação, escala, PCA, balanceamento, remoção de
  linhas) fica **dentro** de um `imblearn.pipeline.Pipeline`, para ser ajustada só na dobra de treino.
- Ponto de partida de cada parte: `build_cleaning_pipeline(...)` da Parte 1 com a configuração
  `LIMPEZA_PADRAO`; troque **apenas** a sua família de decisão e mantenha o resto fixo.
- Registro: cada variante vira uma linha via `run_experiment(...)`; junte com `results_table(...)`.
  Métrica principal: F1-macro (média ± desvio). Também: recall e F1 da classe `anomaly`, tempo.
- Classificador de referência: Árvore de Decisão (`random_state=42`). Quem trabalha com técnicas
  sensíveis à escala (Partes 3 e 5) repete também com um modelo sensível (Regressão Logística/KNN).
- Cada pessoa entrega: notebook `notebooks/02_<parte>.ipynb`, tabela de resultados e um parágrafo
  de justificativa para o relatório.

## Divisão

| Parte | Responsável | Requisito | Variantes mínimas a comparar |
|---|---|---|---|
| **1** | Samuel | Redundantes/inconsistentes + harness comum | colunas suspeitas fora/dentro; categorias normalizadas; inválidos mantidos / → NaN + imputação / clip; duplicatas mantidas/removidas; conflitos mantidos/removidos/voto majoritário |
| **2** | _a definir_ | Conversão simbólico-numérica | one-hot vs. ordinal vs. frequency encoding (e/ou target encoding com validação interna); discutir cardinalidade de `service`, dimensionalidade gerada e categorias desconhecidas no teste |
| **3** | _a definir_ | Normalização/transformação numérica | nenhuma vs. `StandardScaler` vs. `MinMaxScaler` vs. `log1p` + `RobustScaler` (bytes e durações muito assimétricos); mostrar que árvore é invariante e modelo sensível à escala não é |
| **4** | _a definir_ | Balanceamento (2 técnicas) | sem balanceamento vs. `RandomUnderSampler` vs. `SMOTENC` (antes do one-hot) ou SMOTE pós-codificação justificado; opcional `class_weight="balanced"` |
| **5** | _a definir_ | Redução de dimensionalidade ou seleção de instâncias + consolidação | PCA (variância 90/95/99%) **ou** Tomek Links / ENN; depois junta as tabelas das Partes 1–5 no notebook único da etapa e redige a seção de pré-processamento |

### Dependências e ordem

1. Parte 1 publica o harness e a configuração de limpeza padrão (desbloqueia as demais).
2. Partes 2, 3, 4 e 5 rodam em paralelo sobre `LIMPEZA_PADRAO`.
3. Parte 5 consolida: uma tabela única `etapa × variante × F1-macro × recall anomaly × tempo`.
4. Revisão cruzada: 1 revisa 4 · 2 revisa 1 · 3 revisa 2 · 4 revisa 5 · 5 revisa 3. Checar
   vazamento (algo ajustado fora da dobra?) e se a comparação muda uma coisa por vez.

## Armadilhas que derrubam a nota de qualidade metodológica

- Escalar, codificar, fazer PCA ou balancear o `treino.csv` inteiro antes do `cross_validate`.
- Balancear (ou remover duplicatas) também na dobra de validação.
- SMOTE direto sobre one-hot gera categorias fracionárias; preferir `SMOTENC` ou justificar.
- Usar `tipo_alerta_soc`, `score_ids_legado` ou `tempo_resposta_analista_min` no modelo principal
  (ver EDA, seção 9 do relatório da Etapa 1).
- Escolher a variante pelo melhor F1 de uma única dobra; reportar média e desvio.
