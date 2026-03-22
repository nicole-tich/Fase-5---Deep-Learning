# 🎓 Passos Mágicos — Preditor de Risco de Defasagem Escolar

> **FIAP Pós-Graduação em Data Analytics · Fase 5 — Datathon**

Solução completa de análise de dados e Machine Learning para a associação educacional **Passos Mágicos**, com o objetivo de identificar antecipadamente alunos em risco de defasagem escolar com base em indicadores pedagógicos e psicossociais.

---

## 📌 Problema

A Passos Mágicos acompanha anualmente centenas de alunos em situação de vulnerabilidade por meio de indicadores como INDE, IAN, IDA, IEG, IAA, IPS, IPP e IPV. O desafio consiste em:

1. Responder **11 questões de negócio** sobre o perfil e a evolução dos alunos (2020–2024)
2. Construir um **modelo preditivo** para identificar alunos em risco de defasagem (IAN ≤ 5) antes que a situação se agrave
3. Disponibilizar os resultados por meio de um **aplicativo interativo**

---

## 🗂️ Estrutura do Projeto

```
.
├── Datathon_FIAP_Fase_5.ipynb   # Notebook principal: EDA + ML
├── utils_pm.py                  # Módulo de pré-processamento compartilhado
├── app.py                       # Aplicativo Streamlit
├── requirements.txt
├── .streamlit/
│   └── config.toml              # Tema da aplicação
├── dados/
│   └── processado/              # Dados gerados pelo notebook
├── modelo/                      # Artefatos do modelo (gerados pelo notebook)
│   ├── modelo_passos_magicos.pkl
│   ├── config_passos_magicos.pkl
│   └── feature_names.pkl
└── .env                         # Configuração de caminhos
```

---

## ⚙️ Configuração do Ambiente

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
DATA_PATH=dados/PEDE_PASSOS_DATASET_FIAP.xlsx
MODELS=modelo
PROCESSED=dados/processado
```

### 4. Criar pastas necessárias

```bash
mkdir -p dados/processado modelo
```

---

## 📓 Executando o Notebook

Abra `Datathon_FIAP_Fase_5.ipynb` e execute célula a célula:

- **Células 1–9**: Análise Exploratória respondendo as 11 questões de negócio
- **Células 10–15**: Pipeline de Machine Learning, avaliação e exportação dos artefatos

O notebook gera automaticamente os arquivos em `modelo/` necessários para o app.

---

## 🚀 Executando o Aplicativo Localmente

```bash
streamlit run app.py
```

Acesse `http://localhost:8501` no navegador.

---

## ☁️ Deploy no Streamlit Community Cloud

1. Faça push do repositório para o GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Clique em **New app** e conecte ao seu repositório
4. Defina:
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Em **Advanced settings → Secrets**, adicione:
   ```toml
   DATA_PATH = "dados/PEDE_PASSOS_DATASET_FIAP.xlsx"
   MODELS = "modelo"
   PROCESSED = "dados/processado"
   ```
6. Clique em **Deploy**

> **Nota**: Para o app funcionar no Cloud, você precisará incluir os arquivos `.pkl` em `modelo/` ou ajustar o pipeline para treinar na inicialização.

---

## 🧠 Abordagem de Modelagem

| Aspecto | Decisão |
|---|---|
| **Alvo** | `risco_defasagem = 1` se `IAN ≤ 5` |
| **Split temporal** | Treino: anos < 2024 · Teste: ano == 2024 |
| **Modelos avaliados** | Logistic Regression, Random Forest, Gradient Boosting, LightGBM |
| **Pré-processamento** | SimpleImputer (mediana) + StandardScaler · flags de valores faltantes |
| **Features derivadas** | `std_notas`, `risco_psico`, `delta_inde`, `evolucao_inde_pct` |
| **Threshold** | Otimizado por F-beta (β=2) — prioriza recall para não perder alunos em risco |
| **Validação** | StratifiedKFold 5 folds · métricas: ROC AUC, PR AUC, F-beta |

---

## 📊 Questões de Negócio Respondidas

1. **Q1** — Perfil do estudante com defasagem (IAN ≤ 5)
2. **Q2** — Evolução do IDA ao longo dos anos
3. **Q3** — Correlação entre IEG, IDA e IPV
4. **Q4** — Relação entre IAA e desempenho acadêmico
5. **Q5** — Distribuição do IPS por fase
6. **Q6** — Impacto do IPP no risco de defasagem
7. **Q7** — Fatores correlacionados com o IPV
8. **Q8** — Composição multidimensional do INDE
9. *(Q9 explorada dentro de Q8)*
10. **Q10** — Efetividade do programa por classificação Pedra
11. **Q11** — Insights adicionais (gênero, variância de notas, evolução INDE)

---

## 🛠️ Tecnologias Utilizadas

- **Python** 3.9+
- **Pandas / NumPy** — manipulação de dados
- **Scikit-learn** — pipeline de ML
- **LightGBM** — modelo de gradient boosting
- **Matplotlib / Seaborn** — visualizações
- **Streamlit** — aplicativo web interativo
- **Joblib** — serialização de artefatos

---

## ⚠️ Aviso Ético

Este modelo é uma **ferramenta de apoio pedagógico** e não substitui a avaliação qualitativa de educadores e psicólogos. As predições devem ser utilizadas como sinal de alerta inicial, sempre complementadas pelo olhar humano especializado.

---

*Desenvolvido por Nicole Tometich e Giovanni Gerodo · FIAP Pós-Graduação em Data Analytics · 2024*
