# Passos Mágicos — Preditor de Risco de Defasagem Escolar

**FIAP Pos-Graduacao em Data Analytics · Fase 5 — Datathon**

Solucao de analise de dados e Machine Learning para a associacao educacional **Passos Magicos**, com o objetivo de identificar antecipadamente alunos em risco de defasagem escolar com base em indicadores pedagogicos e psicossociais.

---

## Problema

A Passos Magicos acompanha anualmente centenas de alunos em situacao de vulnerabilidade por meio de indicadores como INDE, IAN, IDA, IEG, IAA, IPS, IPP e IPV. O desafio consiste em:

1. Responder **10 questoes de negocio** sobre o perfil e a evolucao dos alunos (2022–2024) - documento anexo ao trabalho
2. Construir um **modelo preditivo** para identificar alunos em risco de defasagem (IAN <= 5) antes que a situacao se agrave
3. Disponibilizar os resultados por meio de um **aplicativo interativo**

---

## Estrutura do Projeto

```
.
├── Datathon_FIAP_Fase_5.ipynb   # Notebook principal: EDA + ML
├── utils_pm.py                  # Modulo de pre-processamento compartilhado
├── app.py                       # Aplicativo Streamlit
├── modelagem_preditiva.md       # Documentacao da abordagem de modelagem
├── requirements.txt
├── .streamlit/
│   └── config.toml              # Tema da aplicacao
├── dados/
│   └── processado/              # Dados e imagens gerados pelo notebook
├── modelo/                      # Artefatos do modelo (gerados pelo notebook)
│   ├── modelo_passos_magicos.pkl
│   ├── config_passos_magicos.pkl
│   └── feature_names.pkl
└── .env                         # Configuracao de caminhos
```

---

## Configuracao do Ambiente

### 1. Clonar o repositorio

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar variaveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
DATA_PATH=dados/PEDE_PASSOS_DATASET_FIAP.xlsx
MODELS=modelo
PROCESSED=dados/processado
```

### 4. Criar pastas necessarias

```bash
mkdir dados/processado modelo
```

---

## Executando o Notebook

Abra `Datathon_FIAP_Fase_5.ipynb` e execute as celulas em ordem:

- **Secao 1**: Setup, carregamento e inspecao dos dados brutos
- **Secao 2**: Analise Exploratoria (EDA) respondendo as 10 questoes de negocio
- **Secao 3**: Engenharia de features e preparacao para modelagem
- **Secao 4**: Treinamento, validacao cruzada e avaliacao dos modelos
- **Secao 5**: Exportacao dos artefatos (`.pkl`) para uso no aplicativo

O notebook gera automaticamente os arquivos em `modelo/` necessarios para o app.

---

## Executando o Aplicativo Localmente

```bash
streamlit run app.py
```

Acesse `http://localhost:8501` no navegador.

O app oferece duas funcionalidades:
- **Predicao Individual**: preenchimento manual dos indicadores de um aluno para obtencao imediata do risco
- **Analise em Lote**: upload de arquivo Excel ou CSV com multiplos alunos para predicao em massa

---

## Deploy no Streamlit Community Cloud

1. Faca push do repositorio para o GitHub (incluindo os arquivos `.pkl` em `modelo/`)
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Clique em **New app** e conecte ao repositorio
4. Defina:
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Em **Advanced settings > Secrets**, adicione:
   ```toml
   DATA_PATH = "dados/PEDE_PASSOS_DATASET_FIAP.xlsx"
   MODELS = "modelo"
   PROCESSED = "dados/processado"
   ```
6. Clique em **Deploy**

---

## Abordagem de Modelagem

| Aspecto | Decisao |
|---|---|
| **Variavel alvo** | `risco_defasagem = 1` se `IAN <= 5`, `0` caso contrario |
| **Divisao treino/teste** | `train_test_split` estratificado — 70% treino, 30% teste |
| **Modelos comparados** | Logistic Regression, Random Forest, SVC (kernel RBF) |
| **Modelo final** | Random Forest (`n_estimators=300`, `class_weight="balanced_subsample"`) |
| **Features** | `genero`, `idade`, `fase_ideal`, `mat`, `por`, `ing`, `iaa`, `ieg`, `ips`, `ipp`, `ida`, `ipv`, `inde_2023`, `n_av`, `media_academica`, `std_notas`, `media_comportamental`, `risco_psico`, `miss_iaa`, `miss_ieg`, `miss_ips`, `miss_ipp`, `miss_ida`, `miss_ipv` (24 features) |
| **Pre-processamento** | `SimpleImputer` (mediana) + `StandardScaler` para numericas; `OneHotEncoder` para categoricas |
| **Threshold de decisao** | Fixo em 0.40 — prioriza identificacao de casos em risco |
| **Zonas de risco** | Alto (>= 40%), Moderado (36-39%), Baixo (< 36%) |
| **Validacao** | `StratifiedKFold` com 5 folds · metricas: ROC AUC, PR AUC, Acuracia |

Para detalhamento completo da metodologia de modelagem, consulte [`modelagem_preditiva.md`](modelagem_preditiva.md).

---

## Tecnologias Utilizadas

- **Python 3.9+**
- **Pandas / NumPy** — manipulacao e transformacao de dados
- **Scikit-learn** — pipeline de pre-processamento e modelagem
- **Matplotlib / Seaborn** — visualizacoes estatisticas
- **Streamlit** — aplicativo web interativo
- **Joblib** — serializacao dos artefatos do modelo

---

## Aviso Etico

Este modelo e uma ferramenta de apoio pedagogico e nao substitui a avaliacao qualitativa de educadores e psicologos. As predicoes devem ser utilizadas como sinal de alerta inicial, sempre complementadas pelo julgamento humano especializado.

---

*Desenvolvido por Nicole Tometich e Giovanni Gerodo · FIAP Pos-Graduacao em Data Analytics · 2026*
