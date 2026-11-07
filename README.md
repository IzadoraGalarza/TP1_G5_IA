# TP1_G5_IA
Repositório para o trabalho prática da disciplina de Inteligência Artificial

## Organização do trabalho

A branch `Develop` contém o notebook original da equipe, `Trabalho_IA_Etapa_1.ipynb`, com a
análise exploratória consolidada. A branch `parte-1-eda-baseline` parte diretamente de `Develop` e
acrescenta validação automatizada, um baseline sem vazamento e o plano de divisão entre cinco
pessoas.

Arquivos principais desta parte:

- `Trabalho_IA_Etapa_1.ipynb`: EDA original da equipe, preservada sem alterações;
- `notebooks/01_eda_baseline.ipynb`: contrato dos arquivos oficiais e baseline;
- `src/data_validation.py`: validação de esquema, IDs, classes e qualidade dos dados;
- `src/baseline.py`: Árvore de Decisão avaliada com pré-processamento dentro das dobras;
- `docs/plano_equipe.md`: riscos, três trilhas de revisão e divisão equilibrada da equipe.

## Execução local

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

O notebook original espera `base_nsl_kdd.csv`, mas esse arquivo não está versionado na branch
`Develop`. O notebook complementar espera os arquivos oficiais da competição dentro de `data/`:

```text
data/
├── treino.csv
├── teste_sem_rotulo.csv
└── modelo_de_submissao.csv
```

## Entrega de 07/11 - Etapa 2

O arquivo principal desta entrega é `Trabalho_IA_Etapa_2.ipynb`. Ele cobre somente o escopo de
pré-processamento solicitado para 07/11:

- comparação de tratamentos de redundâncias e inconsistências;
- `RandomUnderSampler` versus `RandomOverSampler`, além do controle sem balanceamento;
- one-hot encoding com categorias desconhecidas ignoradas;
- imputação, indicadores de ausência e padronização numérica;
- redução de dimensionalidade com Truncated SVD;
- validação cruzada sem embaralhamento, mantendo assinaturas idênticas na mesma dobra;
- exportação das tabelas para `output/etapa_2/`.

Para gerar os resultados reais, adicione `base_nsl_kdd.csv` na raiz do projeto ou em `data/` e
execute todas as células do notebook. O relatório da Etapa 1 não contém as linhas da base e, por
isso, não substitui o CSV na execução dos experimentos.
