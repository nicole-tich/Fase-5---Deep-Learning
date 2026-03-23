# Modelagem Preditiva — Identificacao de Risco de Defasagem

## Objetivo

Construir um classificador binario capaz de identificar, com base nos indicadores disponiveis, se um aluno apresenta risco de defasagem escolar. A variavel alvo e definida como:

- `risco_defasagem = 1`: aluno com IAN <= 5 (alto risco)
- `risco_defasagem = 0`: demais casos

O foco e na antecipacao do risco, por isso, priorizamos modelos com boa capacidade de identificar casos positivos mesmo que ao custo de alguns falsos positivos.

---

## Preparacao dos Dados

### Features utilizadas

O conjunto de features e definido explicitamente na lista `FEATURES_MODELO` em `utils_pm.py`, garantindo consistencia exata entre treino e inferencia no app. As features sao:

| Grupo | Features |
|---|---|
| Dados do aluno | `genero`, `idade`, `fase_ideal` |
| Notas academicas | `mat`, `por`, `ing` |
| Indicadores pedagogicos/psicossociais | `iaa`, `ieg`, `ips`, `ipp`, `ida`, `ipv` |
| Historico INDE | `inde_2023` |
| Outros | `n_av` |
| Derivadas | `media_academica`, `std_notas`, `media_comportamental`, `risco_psico` |
| Flags de ausencia | `miss_iaa`, `miss_ieg`, `miss_ips`, `miss_ipp`, `miss_ida`, `miss_ipv` |

As colunas `inde_2022`, `delta_inde` e `evolucao_inde_pct` foram excluidas porque nao estao presentes no dataset consolidado — elas so existiriam se todos os anos tivessem a mesma coluna de INDE, o que nao ocorre. Inclui-las causaria NaN sistematico na inferencia e corromperia as predicoes.

As flags `miss_*` registram se cada indicador estava ausente no registro original, distinguindo um valor baixo (ex: IPS=3) de um valor nunca medido (NaN). Isso e relevante porque alunos sem avaliacao registrada podem ter perfil de risco distinto dos que foram avaliados e tiveram resultado ruim.

Colunas do tipo `datetime` sao removidas antes do treinamento por incompatibilidade com o pipeline do scikit-learn.

### Divisao treino/teste

O conjunto de dados foi dividido com `train_test_split` estratificado, utilizando 70% para treino e 30% para teste. A estratificacao garante que a proporcao de positivos (`risco = 1`) seja preservada em ambos os subconjuntos, independentemente da distribuicao temporal dos dados.

Essa abordagem foi preferida ao split temporal (ex: treinar em 2022-2023 e testar em 2024) porque o volume de dados por ano e limitado e o split temporal tenderia a concentrar toda a diversidade de perfis do dataset mais recente no conjunto de teste, reduzindo a representatividade do treino.

---

## Pipeline de Pre-processamento

O pre-processamento foi implementado via `ColumnTransformer` com dois ramos:

- **Numericas**: imputacao pela mediana (`SimpleImputer`) seguida de normalizacao (`StandardScaler`)
- **Categoricas**: imputacao pela moda e codificacao com `OneHotEncoder` (ignora categorias desconhecidas)

O preprocessador e ajustado exclusivamente no conjunto de treino e aplicado ao de teste.

---

## Modelos Comparados

Foram avaliados tres classificadores:

| Modelo | Configuracao Principal |
|---|---|
| Logistic Regression | `max_iter=2000`, `class_weight="balanced"` |
| Random Forest | `n_estimators=300`, `class_weight="balanced_subsample"` |
| SVC (kernel RBF) | `class_weight="balanced"`, `probability=True` |

O parametro `class_weight="balanced"` foi utilizado em todos os modelos para compensar o desbalanceamento entre as classes, atribuindo maior peso aos casos de risco.

---

## Validacao Cruzada

Antes da avaliacao final, cada modelo passou por validacao cruzada estratificada com 5 folds sobre o conjunto de treino. As metricas avaliadas foram:

- **ROC AUC**: métrica geral de separação entre as duas classes (alunos em risco vs. sem risco). Varia de 0 a 1, onde 1 indica perfeita classificação
- **Precisão**: entre os alunos preditos como em risco, quantos realmente estão em risco (reduz falsos alarmes)
- **Recall**: de todos os alunos que realmente estão em risco, quantos o modelo consegue identificar (evita deixar alunos em risco sem identificar)
- **F1-Score**: equilíbrio entre precisão e recall, útil para dados desbalanceados como este

A validacao cruzada serve para estimar a estabilidade do modelo e detectar overfitting antes da avaliacao no conjunto de teste.

---

## Avaliacao no Conjunto de Teste

Apos a validacao cruzada, cada modelo foi retreinado com todo o conjunto de treino e avaliado no conjunto de teste (30% dos dados). As metricas reportadas incluem ROC AUC, relatorio de classificacao por classe e matriz de confusao.

O modelo com maior ROC AUC no teste e selecionado automaticamente como modelo final.

---

## Modelo Final: Logistic Regression

A Regressao Logistica foi o modelo com melhor desempenho geral no conjunto de teste, e e tambem o mais adequado para este contexto pelas razoes abaixo.

### Por que Logistic Regression

**Interpretabilidade.** Os coeficientes fornecem uma medida direta do impacto de cada indicador sobre o risco de defasagem, permitindo que educadores entendam e justifiquem as predições com base em evidências quantificáveis.

**Robustez com Dados Limitados.** Com centenas de registros por ano, modelos complexos (Random Forest, SVC) tendem a overfitting. A Regressão Logística, sendo linear, oferece melhor generalização e menor variância em amostras reduzidas.

**Eficiência e Calibração.** Alcança desempenho equivalente aos modelos complexos no teste, com tempo de treinamento negligenciável e output naturalmente interpretado como probabilidade, facilitando ajuste do threshold e deployment em ambientes com recursos limitados.


---

## Threshold de Decisao e Zonas de Risco

O threshold foi fixado em **0.40** (ao inves do padrao de 0.50). Isso significa que um aluno e classificado como em risco quando a probabilidade estimada pelo modelo e maior ou igual a 40%.

A reducao do threshold aumenta o recall da classe positiva — ou seja, o modelo identifica mais alunos em risco — ao custo de aumentar os falsos positivos (alunos sem risco classificados como em risco). Em um contexto de intervencao pedagogica preventiva, o custo de se perder um aluno em risco (falso negativo) e maior do que o custo de uma intervencao desnecessaria (falso positivo). Por isso, o threshold conservador e justificado.

O app classifica as predicoes em tres zonas:

| Zona | Faixa de Probabilidade | Acao Sugerida |
|---|---|---|
| Alto Risco | >= 40% | Intervencao pedagogica e acompanhamento psicossocial intensificados |
| Risco Moderado | 36% a 39% (dentro de 10% do limiar) | Monitoramento proximo nas proximas avaliacoes |
| Baixo Risco | < 36% | Acompanhamento padrao |

A zona moderada e propositalmente estreita (4 pontos percentuais) para evitar falsos alarmes em alunos claramente saudaveis. O uso de `class_weight="balanced"` faz com que a probabilidade base do modelo seja deslocada para cima em relacao a 50%, e a escala interpretavel e 0-40% (baixo) e nao 0-50%.

---

## Artefatos Exportados

Ao final do notebook, tres arquivos sao salvos na pasta `modelo/`:

| Arquivo | Conteudo |
|---|---|
| `modelo_passos_magicos.pkl` | Pipeline completo (preprocessador + classificador) |
| `config_passos_magicos.pkl` | Nome do modelo escolhido e threshold de decisao |
| `feature_names.pkl` | Lista de features esperadas na entrada do modelo |

Esses artefatos sao carregados diretamente pelo aplicativo Streamlit (`app.py`) para realizar predicoes sem necessidade de re-treinamento.
