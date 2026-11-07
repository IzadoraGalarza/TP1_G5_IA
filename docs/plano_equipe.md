# Plano de execução e revisão

## Leitura executiva do trabalho

O objetivo não é apenas maximizar o F1-macro. A equipe precisa demonstrar uma comparação controlada
de pré-processamentos e de pelo menos três classificadores, impedir vazamento de dados, gerar uma
submissão reproduzível e explicar o modelo escolhido. Os entregáveis finais são um notebook completo,
um relatório de 3 a 5 páginas e `submissao_<numero_da_equipe>.csv`.

## Estado encontrado na branch `Develop`

A equipe já produziu `Trabalho_IA_Etapa_1.ipynb`, com uma EDA extensa e resultados consolidados
sobre uma base de 69.833 linhas e 50 colunas. Esse trabalho deve ser reaproveitado, mas ainda há
quatro pontos a resolver antes de tratá-lo como resultado reproduzível:

1. `base_nsl_kdd.csv` não está versionado e o notebook depende desse caminho externo;
2. a base analisada contém atributos enriquecidos que não pertencem ao esquema canônico de 41
   preditores, portanto sua proveniência precisa ser registrada;
3. números consolidados permanecem escritos no texto mesmo quando o arquivo não é carregado;
4. ainda não há pipeline de modelagem, validação cruzada nem geração da submissão oficial.

Por isso, a EDA existente será mantida como evidência da Etapa 1, enquanto os módulos em `src/` e o
notebook complementar estabelecem o contrato reproduzível para as etapas seguintes.

## Maiores problemas e riscos

1. **Dados ainda ausentes no projeto.** Sem os três CSVs oficiais não é possível confirmar esquema,
   volume, distribuição, redundâncias nem resultados.
2. **Prazo imediato.** A etapa de pré-processamento está marcada para 07/11, enquanto a etapa de EDA
   aparece como 01/11. É preciso produzir primeiro uma execução mínima reproduzível e depois ampliar
   a matriz experimental.
3. **Regra ambígua sobre divisão dos dados.** O enunciado proíbe “re-embaralhar ou criar um split
   próprio”, mas também manda selecionar o modelo por validação cruzada ou split dentro do treino.
   A decisão conservadora é usar `StratifiedKFold(shuffle=False)` e documentar essa interpretação.
4. **Vazamento no pré-processamento.** Imputação, codificação, normalização, seleção e balanceamento
   precisam ser ajustados apenas na parte de treino de cada dobra. Fazer qualquer etapa antes da
   validação cruzada invalida a estimativa.
5. **SMOTE e atributos categóricos.** Aplicar SMOTE diretamente depois de one-hot pode criar vetores
   categóricos fracionários pouco realistas. Comparar `RandomUnderSampler` com `SMOTENC` antes da
   codificação, ou justificar explicitamente uma alternativa compatível.
6. **Alta cardinalidade de `service`.** One-hot pode aumentar muito a dimensionalidade e deixar
   categorias raras instáveis. Medir cardinalidade, categorias desconhecidas e custo do modelo.
7. **Possíveis atributos de vazamento.** `id`, rótulo, nível de dificuldade ou derivados não devem
   entrar automaticamente no modelo. Correlação isolada não prova vazamento; origem e disponibilidade
   do atributo no momento da inferência também devem ser analisadas.
8. **Desbalanceamento.** Acurácia não orienta a seleção. Reportar F1-macro, precisão e recall por
   classe, além de matriz de confusão fora da amostra.
9. **Explosão combinatória.** Testar todas as combinações é caro e dificulta interpretar causa e
   efeito. Usar baseline fixo, trocar uma família de decisão por vez e só então combinar as melhores.
10. **Sobreajuste à validação.** Muitas tentativas no mesmo esquema de CV podem otimizar ruído.
    Congelar dobras e sementes, registrar todas as tentativas e preferir modelos estáveis.
11. **Explicabilidade inconsistente.** Importância por impureza pode favorecer atributos contínuos ou
    de alta cardinalidade. Preferir permutation importance na validação e complementar com a explicação
    específica do modelo.
12. **Submissão rígida.** IDs, ordem, quantidade de linhas, valores permitidos e ausência de nulos
    devem ser validados automaticamente antes da entrega.
13. **Reprodutibilidade.** Fixar sementes, versões, caminhos relativos e fazer o notebook executar do
    início ao fim em ambiente limpo.
14. **Integridade acadêmica.** Declarar no relatório o uso de IA para planejamento, redação assistida
    e depuração, conforme solicitado no enunciado.

## Três trilhas de revisão

As trilhas abaixo são resumos verificáveis de raciocínio, não registros de pensamento privado.

### Trilha 1 - validade metodológica

`arquivos oficiais -> contrato e alvo -> dobras fixas -> transformações dentro da dobra -> balanceamento
apenas no treino -> avaliação fora da amostra -> ajuste final apenas no treino completo`

Checklist:

- o teste sem rótulo nunca participa de ajuste ou seleção;
- `id`, alvo e atributos indisponíveis na inferência ficam fora de `X`;
- todas as etapas aprendidas estão dentro do pipeline;
- o balanceador é executado somente nas dobras de treino;
- não há deduplicação cruzada usando informação do conjunto de teste;
- o modelo final é escolhido somente por evidência do treino.

### Trilha 2 - qualidade experimental

`baseline -> variações isoladas -> validação idêntica -> métricas por dobra e classe -> estabilidade ->
combinação vencedora -> confirmação`

Checklist:

- mesma definição de dobras para todas as comparações;
- baseline sem balanceamento registrado antes das melhorias;
- pelo menos duas técnicas de balanceamento e três classificadores;
- média e desvio-padrão do F1-macro, não apenas o melhor valor;
- tempo e complexidade registrados quando forem relevantes;
- tabela única com pré-processamento, algoritmo e métricas.

### Trilha 3 - reprodução e entrega

`ambiente limpo -> notebook inteiro -> artefatos versionados -> explicação -> refit final -> previsão ->
validação do CSV -> relatório e declaração de IA`

Checklist:

- notebook roda do início ao fim sem estado oculto;
- versões e sementes estão registradas;
- tabela e figuras são geradas pelo próprio código;
- submissão preserva exatamente os IDs e a ordem do teste;
- `class_prevista` contém somente `normal` ou `anomaly`;
- relatório cabe em 3 a 5 páginas e referencia os experimentos reais.

## Divisão equilibrada entre cinco pessoas

Cada pessoa entrega um pacote com cinco itens de peso semelhante: implementação, experimento,
evidência/tabela, texto do relatório e revisão cruzada. Os nomes podem substituir os rótulos abaixo.

| Pessoa | Implementação e experimento principal | Evidência e texto | Revisão cruzada | Peso |
|---|---|---|---|---:|
| **1 (Samuel)** | contrato dos CSVs, EDA, auditoria de vazamento e baseline Árvore de Decisão | diagnóstico dos dados e seção de riscos metodológicos | revisa a trilha 3 da Pessoa 5 | 20% |
| **2** | tratamento de ausentes, duplicatas e inconsistências; comparação de codificação; Regressão Logística | tabela de limpeza/codificação e justificativa de `service` | revisa a trilha 1 da Pessoa 4 | 20% |
| **3** | transformação numérica e redução/seleção de atributos; terceiro classificador (MLP ou KNN) | tabela de dimensionalidade, custo e desempenho | revisa a trilha 2 da Pessoa 2 | 20% |
| **4** | `RandomUnderSampler` versus SMOTE compatível com dados mistos; Random Forest | métricas por classe e análise de estabilidade | revisa a trilha 1 da Pessoa 1 | 20% |
| **5** | matriz experimental integrada, explicabilidade, refit final e geração/validação da submissão | tabela final, conclusão e montagem do relatório | revisa a trilha 2 da Pessoa 3 | 20% |

### Critério de conclusão individual

Uma parte só está concluída quando tem: código executável, resultado reproduzível, pelo menos uma
comparação justa, texto curto para o relatório e revisão por outra pessoa. A integração é coletiva,
mas a Pessoa 5 mantém o pipeline final para evitar cinco versões divergentes.

## Ordem recomendada

1. Pessoa 1 valida os arquivos e congela o contrato e as dobras.
2. Pessoas 2, 3 e 4 trabalham em paralelo sobre o mesmo contrato.
3. Pessoa 5 agrega resultados em uma tabela padronizada.
4. Cada pessoa executa sua revisão cruzada e registra achados.
5. A equipe fecha o modelo, explicabilidade, submissão e relatório.
