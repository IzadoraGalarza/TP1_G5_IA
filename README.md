# TP1_G5_IA

Trabalho em competição da disciplina de Inteligência Artificial (Profª Cristiane Neri Nobre):
detecção de intrusão em redes com a base NSL-KDD. Métrica oficial: F1-macro no conjunto de teste.

O notebook principal da equipe fica no Colab do grupo. Este repositório guarda a base, as entregas e
as partes individuais.

## Etapas

| Etapa | Prazo | Situação |
|---|---|---|
| 1. Análise exploratória | 01/11 | ✅ entregue (`Trabalho_IA_Etapa_1.ipynb` e relatório em PDF) |
| 2. Pré-processamento | 07/11 | 🟡 em andamento |
| 3. Classificadores | 21/11 | ⚪ |
| 4. Competição e relatório final | 28/11 | ⚪ |

## Etapa 2: divisão

| Parte | Responsável | Situação |
|---|---|---|
| 1. Redundantes/inconsistentes + estrutura comum | Izadora | ✅ [resumo](docs/parte1_redundantes.md) · [notebook](-) · [abrir no Colab](https://colab.research.google.com/drive/1ANn4rmYFsz3ziuuhk25IlDa1llmk6GqQ?usp=sharing) |  |
| 2. Codificação categórica | Samuel | ✅ [resumo](docs/parte2_codificacao.md) · [notebook](notebooks/parte2_codificacao.ipynb) · [abrir no Colab](https://colab.research.google.com/github/IzadoraGalarza/TP1_G5_IA/blob/main/notebooks/parte2_codificacao.ipynb) |
| 3. Normalização | Izadora | ✅ [resumo](docs/parte3_normalizacao.md) · [notebook](-) · [abrir no Colab](https://colab.research.google.com/drive/1ANn4rmYFsz3ziuuhk25IlDa1llmk6GqQ?usp=sharing) | 
| 4. Balanceamento | a definir | — |
| 5. PCA ou seleção de instâncias + tabela final | a definir | — |

Regra de ouro: tudo o que aprende algo dos dados (imputação, codificação, escala, balanceamento, PCA)
fica dentro do Pipeline e é ajustado só no treino de cada fold. O teste nunca participa de nenhum `fit`.

## Estrutura

```text
├── data/                     # treino.csv, teste_sem_rotulo.csv, modelo_de_submissao.csv (Canvas)
├── docs/                     # resumo de cada parte para a equipe
├── notebooks/                # notebook de cada parte (roda sozinho no Colab)
├── output/etapa_2/           # tabelas e gráficos gerados pelos notebooks
├── Trabalho_IA_Etapa_1.ipynb # EDA entregue na Etapa 1
├── Relatorio_Analise_Exploratoria_NSL_KDD_Etapa_1_entrega.PDF
└── Enunciado_Trabalho_Competicao_IA_23-10-26.pdf
```

Para adicionar a sua parte, siga o mesmo padrão: `notebooks/parteN_<tema>.ipynb`,
`docs/parteN_<tema>.md` e as saídas em `output/etapa_2/`.

## Como executar

**No Colab:** abra o notebook pelo link e use *Ambiente de execução → Executar tudo*. A primeira célula
clona este repositório para ter a base.

**Localmente:**

```bash
git clone https://github.com/IzadoraGalarza/TP1_G5_IA.git
cd TP1_G5_IA
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```
