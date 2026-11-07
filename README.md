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
