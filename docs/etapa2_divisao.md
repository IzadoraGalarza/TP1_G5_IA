# Etapa 2 - divisão equilibrada entre cinco pessoas

Entrega de 07/11: notebook com experimentos de pré-processamento, resultados comparativos e
justificativas. A divisão abaixo considera apenas a Etapa 2 e distribui aproximadamente 20% do
trabalho para cada integrante.

## Protocolo comum

Todas as partes devem usar o mesmo protocolo para que os resultados possam ser reunidos em uma
única tabela:

- apenas a base de treino; o teste sem rótulo não participa dos experimentos;
- `StratifiedGroupKFold` com 5 dobras, sem embaralhamento;
- assinaturas idênticas mantidas na mesma dobra;
- F1-macro como métrica principal, acompanhado de precisão e recall da classe `anomaly`;
- toda imputação, codificação, escala, redução e reamostragem dentro do pipeline;
- Regressão Logística como classificador de referência, sem tratá-la como modelo final;
- sementes fixadas em 42;
- cada comparação altera somente uma família de decisão.

Configuração de controle para as comparações:

- excluir `tipo_alerta_soc`, `score_ids_legado` e `tempo_resposta_analista_min`;
- remover constante e representações derivadas redundantes;
- imputação por moda/mediana;
- one-hot encoding;
- StandardScaler;
- sem balanceamento e sem redução dimensional, exceto nas partes que avaliam essas técnicas.

## Divisão

### 1. Samuel Ferreira Alves Vieira - redundâncias e inconsistências

Responsabilidades:

- validar colunas, classe e atributos com possível vazamento;
- comparar valores inválidos mantidos, convertidos em `NaN` e limitados por `clip`;
- comparar duplicatas mantidas versus removidas somente na dobra de treino;
- manter conflitos de rótulo sem relabelar automaticamente;
- justificar remoção de constante e redundâncias de bytes.

Entrega individual:

- seção de limpeza do notebook;
- tabela `resultados_limpeza.csv`;
- um parágrafo sobre redundâncias/inconsistências;
- pelo menos um teste automatizado.

### 2. Daniel Alves Goncalves - dados ausentes e codificação categórica

Responsabilidades:

- comparar imputação categórica por moda e categoria explícita `ausente`;
- comparar one-hot, ordinal e codificação por frequência;
- medir a dimensionalidade criada por cada alternativa;
- discutir especificamente a cardinalidade de `service` e categorias desconhecidas;
- indicar a codificação mais segura para a configuração comum.

Entrega individual:

- seção de ausentes/codificação do notebook;
- tabela `resultados_codificacao.csv`;
- um gráfico de dimensionalidade por técnica;
- um parágrafo de justificativa e um teste automatizado.

### 3. Daniel Pedrosa Collina - transformação numérica

Responsabilidades:

- comparar sem escala, StandardScaler, MinMaxScaler e RobustScaler;
- aplicar `log1p` somente a atributos não negativos e assimétricos quando justificável;
- manter imputação numérica dentro das dobras;
- registrar o efeito das transformações no F1-macro e no tempo;
- indicar a transformação padrão para a configuração comum.

Entrega individual:

- seção de transformação numérica do notebook;
- tabela `resultados_transformacao.csv`;
- um gráfico comparativo das métricas;
- um parágrafo de justificativa e um teste automatizado.

### 4. Izadora Galarza Alves - balanceamento

Responsabilidades:

- comparar sem balanceamento, RandomUnderSampler e RandomOverSampler;
- executar o balanceamento somente nas dobras de treino;
- registrar a distribuição das classes antes e depois de cada técnica;
- comparar F1-macro, precisão e recall da anomalia;
- discutir perda de informação no undersampling e sobreajuste no oversampling.

Se houver tempo, SMOTENC pode ser incluído como experimento adicional, mas não substitui as duas
técnicas já reproduzíveis. SMOTE direto depois de one-hot não deve ser usado sem justificativa.

Entrega individual:

- seção de balanceamento do notebook;
- tabela `resultados_balanceamento.csv`;
- gráfico de precisão versus recall da anomalia;
- um parágrafo de justificativa e um teste automatizado.

### 5. Larissa Varella Araújo - redução de dimensionalidade

Responsabilidades:

- comparar sem redução e Truncated SVD com 10, 20 e 40 componentes;
- registrar variância explicada acumulada, tempo e F1-macro;
- verificar se a redução preserva o desempenho da classe minoritária;
- indicar o número de componentes com melhor equilíbrio entre desempenho e custo;
- organizar a ordem final das seções no notebook integrado.

A organização final não significa escrever ou corrigir as partes das outras pessoas: cada responsável
entrega sua própria célula, tabela e justificativa prontas para inclusão.

Entrega individual:

- seção de redução dimensional do notebook;
- tabela `resultados_reducao.csv`;
- gráfico de componentes versus variância/F1;
- um parágrafo de justificativa e um teste automatizado.

## Integração também dividida

| Atividade final | Responsável |
|---|---|
| Conferir carregamento, alvo e colunas proibidas | Samuel |
| Conferir se todos os requisitos do enunciado aparecem no notebook | Daniel Alves |
| Reproduzir o ambiente e executar os testes | Daniel Pedrosa |
| Consolidar as cinco tabelas em uma tabela comparativa | Izadora |
| Ordenar as seções e executar “Run all” no notebook final | Larissa |

## Revisão cruzada

- Samuel revisa balanceamento de Izadora;
- Daniel Alves revisa limpeza de Samuel;
- Daniel Pedrosa revisa codificação de Daniel Alves;
- Izadora revisa redução de Larissa;
- Larissa revisa transformação numérica de Daniel Pedrosa.

Cada revisão verifica três pontos: vazamento, comparação justa e coerência entre código, tabela e
justificativa.

## Critério de conclusão de cada parte

Cada pessoa entrega exatamente cinco itens:

1. código executável dentro do pipeline;
2. pelo menos três variantes comparáveis, incluindo o controle;
3. tabela CSV de resultados;
4. uma figura e um parágrafo curto para o notebook;
5. um teste e uma revisão cruzada registrada.

Assim, nenhuma pessoa fica apenas com redação ou apenas com integração, e todas contribuem com
implementação, evidência e justificativa.
