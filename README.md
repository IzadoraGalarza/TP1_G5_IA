# TP1_G5_IA

Trabalho em competição da disciplina de Inteligência Artificial (Profª Cristiane Neri Nobre):
detecção de intrusão em redes com a base NSL-KDD. Métrica oficial: F1-macro no conjunto de teste.

> **Começando agora?** Leia [`docs/STATUS_ETAPA2.md`](docs/STATUS_ETAPA2.md): o que já está pronto,
> resultados obtidos e o que falta para cada pessoa.

## Etapas

| Etapa | Prazo | Situação | Arquivos |
|---|---|---|---|
| 1. Análise exploratória | 01/11 | ✅ entregue | `Trabalho_IA_Etapa_1.ipynb`, `Relatorio_Analise_Exploratoria_NSL_KDD_Etapa_1_entrega.PDF`, `notebooks/01_eda_baseline.ipynb` |
| 2. Pré-processamento | 07/11 | 🟡 em andamento | `Trabalho_IA_Etapa_2.ipynb`, `notebooks/02_codificacao.ipynb`, `docs/etapa2_divisao.md` |
| 3. Classificadores | 21/11 | ⚪ | — |
| 4. Competição e relatório final | 28/11 | ⚪ | — |

## Etapa 2: divisão

| Parte | Responsável | Situação | Código | Notebook |
|---|---|---|---|---|
| 1. Redundâncias e inconsistências | Daniel Alves | 🟡 rascunho executado | `src/cleaning.py` | `Trabalho_IA_Etapa_2.ipynb` |
| 2. Ausentes e codificação categórica | Samuel | ✅ concluída | `src/encoding.py` | [`notebooks/02_codificacao.ipynb`](https://colab.research.google.com/github/IzadoraGalarza/TP1_G5_IA/blob/main/notebooks/02_codificacao.ipynb) |
| 3. Transformação numérica | Daniel Pedrosa | ⚪ não iniciada | — | — |
| 4. Balanceamento | Izadora | 🟡 rascunho executado | `src/preprocessing_stage2.py` | `Trabalho_IA_Etapa_2.ipynb` |
| 5. Redução de dimensionalidade | Larissa | 🟡 rascunho executado | `src/preprocessing_stage2.py` | `Trabalho_IA_Etapa_2.ipynb` |

Protocolo comum: somente `data/treino.csv`; `StratifiedGroupKFold` com 5 dobras, sem embaralhar e com
assinaturas idênticas na mesma dobra; Regressão Logística de referência; toda imputação, codificação,
escala, redução e reamostragem dentro do pipeline. Detalhes em [`docs/etapa2_divisao.md`](docs/etapa2_divisao.md).

## Estrutura

```text
├── data/                      # CSVs oficiais do Canvas (versionados)
├── docs/
│   ├── STATUS_ETAPA2.md       # o que está feito e o que falta
│   ├── etapa2_divisao.md      # divisão da Etapa 2 entre as cinco pessoas
│   └── plano_equipe.md        # riscos metodológicos e plano geral do trabalho
├── notebooks/                 # notebooks por parte
├── output/etapa_2/            # tabelas (resultados_<parte>.csv) e figuras geradas pelos notebooks
├── src/
│   ├── data_validation.py     # contrato dos CSVs oficiais
│   ├── preprocessing_stage2.py# preparação comum, pré-processador de controle, rascunhos das partes 4 e 5
│   ├── experiments.py         # métricas compartilhadas
│   ├── cleaning.py            # parte 1
│   ├── encoding.py            # parte 2
│   └── baseline.py            # baseline da Etapa 1 (Árvore de Decisão)
└── tests/                     # testes automatizados (unittest)
```

## Como executar

### No Google Colab

Abra o notebook pelo link (ou pelo botão “Open in Colab” no topo dele) e use **Ambiente de execução →
Executar tudo**. A primeira célula clona este repositório, com código e base, e instala as versões
testadas das bibliotecas. Para guardar alterações, use **Arquivo → Salvar uma cópia no Drive**.

- Parte 2 (codificação): [abrir no Colab](https://colab.research.google.com/github/IzadoraGalarza/TP1_G5_IA/blob/main/notebooks/02_codificacao.ipynb)

### Localmente

```bash
git clone https://github.com/IzadoraGalarza/TP1_G5_IA.git
cd TP1_G5_IA
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

Os três arquivos oficiais do Canvas já estão versionados em `data/`:

```text
data/
├── treino.csv
├── teste_sem_rotulo.csv
└── modelo_de_submissao.csv
```

Para conferir os arquivos, rode `python3 -m src.data_validation --data-dir data`. Depois execute os
notebooks do início ao fim. O conjunto de teste sem rótulo nunca é usado para ajustar nada.

### Adicionando a sua parte

Siga o modelo da Parte 2: código em `src/<parte>.py`, notebook em `notebooks/02_<parte>.ipynb`
(copie a primeira célula de setup para ele também abrir no Colab), tabela em
`output/etapa_2/resultados_<parte>.csv` e um teste em `tests/`. Detalhes em
[`docs/STATUS_ETAPA2.md`](docs/STATUS_ETAPA2.md).
