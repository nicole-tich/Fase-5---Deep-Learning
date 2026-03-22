"""
utils_pm.py — Utilitários compartilhados: Datathon Passos Mágicos (Fase 5)
Contém funções de pré-processamento reutilizadas pelo notebook e pelo app Streamlit.
"""
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ─── Normalização de colunas ──────────────────────────────────────────────────

def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas: minúsculas, sem acentos, espaços→underscore."""
    cols = []
    for c in df.columns.astype(str):
        # Remove acentos (NFD decomposition + remove combining marks)
        c = ''.join(ch for ch in unicodedata.normalize('NFD', c)
                    if unicodedata.category(ch) != 'Mn')
        c = c.strip().lower()
        c = re.sub(r'[\s\-\.]+', '_', c)   # espaços/hifens/pontos → _
        c = re.sub(r'[^a-z0-9_]', '', c)    # remove demais caracteres
        c = re.sub(r'_+', '_', c).strip('_')  # colapsa múltiplos _
        cols.append(c)
    df.columns = cols
    return df


# ─── Limpeza / padronização ───────────────────────────────────────────────────

def padronizar_genero_col(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.lower()
    return s.replace({"menino": "masculino", "menina": "feminino"}).where(
        s.isin(["masculino", "feminino", "menino", "menina"])
    )


def extrair_fase_col(s: pd.Series) -> pd.Series:
    def _parse(v):
        if pd.isna(v):
            return np.nan
        v = str(v).lower()
        if "alfa" in v:
            return 0.0
        m = re.search(r"fase\s*(\d+)", v)
        return float(m.group(1)) if m else np.nan
    return s.apply(_parse)


def coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce"
    )


def limpar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza básica: tipos, gênero, idade, fase_ideal, INDE."""
    d = df.copy()
    d = normalizar_colunas(d)  # garante nomes lowercase sem acentos

    if "genero" in d.columns:
        d["genero"] = padronizar_genero_col(d["genero"])

    if "idade" in d.columns:
        dt = pd.to_datetime(d["idade"], errors="coerce")
        d["idade"] = coerce_num(d["idade"])
        mask = dt.notna() & (dt.dt.year == 1900) & (dt.dt.month == 1)
        d.loc[mask, "idade"] = dt[mask].dt.day
        d["idade"] = d["idade"].where(d["idade"].between(6, 30))

    if "fase_ideal" in d.columns:
        d["fase_ideal"] = extrair_fase_col(d["fase_ideal"])

    for col in ["ian", "inde_2022", "inde_2023", "inde_2024",
                "iaa", "ieg", "ida", "ips", "ipp", "ipv",
                "mat", "por", "ing", "n_av"]:
        if col in d.columns:
            d[col] = coerce_num(d[col])

    if "inde_2024" in d.columns:
        tmp = d["inde_2024"].astype(str).str.strip().str.upper()
        d["inde_2024"] = pd.to_numeric(tmp.replace("INCLUIR", np.nan), errors="coerce")

    return d


# ─── Feature engineering ─────────────────────────────────────────────────────

def criar_features_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas sobre o DataFrame já limpo.
    Não remove colunas — apenas adiciona novas.
    """
    d = df.copy()

    # Média acadêmica
    cols_acad = [c for c in ["mat", "por", "ing"] if c in d.columns]
    if len(cols_acad) >= 2:
        d["media_academica"] = d[cols_acad].mean(axis=1)
        # Dispersão das notas (aluno desequilibrado em matérias)
        d["std_notas"] = d[cols_acad].std(axis=1)

    # Média comportamental
    cols_comp = [c for c in ["iaa", "ieg", "ips", "ipp"] if c in d.columns]
    if len(cols_comp) >= 2:
        d["media_comportamental"] = d[cols_comp].mean(axis=1)

    # Evolução INDE
    if "inde_2022" in d.columns and "inde_2023" in d.columns:
        d["delta_inde"] = d["inde_2023"] - d["inde_2022"]
        ref = d["inde_2022"].replace(0, np.nan)
        d["evolucao_inde_pct"] = (d["delta_inde"] / ref) * 100

    # Score de risco psicossocial (0-10, quanto maior maior o risco)
    cols_psico = [c for c in ["ips", "ipp"] if c in d.columns]
    if len(cols_psico) >= 1:
        d["risco_psico"] = 10 - d[cols_psico].mean(axis=1)

    # Flags de missing em indicadores críticos
    for col in ["iaa", "ieg", "ips", "ipp", "ida", "ipv"]:
        if col in d.columns:
            d[f"miss_{col}"] = d[col].isna().astype(int)

    return d


# ─── Preparação completa ──────────────────────────────────────────────────────

COLUNAS_REMOVER = [
    "ra", "nome", "data_nasc", "escola",
    "avaliador_1", "avaliador_2", "avaliador_3",
    "avaliador_4", "avaliador_5", "avaliador_6",
    "rec_av1", "rec_av2", "rec_av3",
    "rec_av4", "rec_av5", "rec_av6",
    "rec_psicologia", "indicado",
    "ativo_inativo", "cg", "cf", "ct",
    "inde_2024", "ano_ingresso",
    "pedra_2020", "pedra_2021", "pedra_2022", "pedra_2023", "pedra_2024",
    "fase", "turma", "instituicao_ensino",
    "atingiu_pv", "destaque_ieg", "destaque_ida", "destaque_ivp",
]


def preparar_features(df: pd.DataFrame, modo_treino: bool) -> pd.DataFrame:
    """
    Pipeline completo de preparação de features.
    modo_treino=True  → cria coluna target 'risco_defasagem'
    modo_treino=False → não cria target (uso no app)
    """
    d = limpar_base(df)
    d = criar_features_derivadas(d)

    if modo_treino:
        if "ian" not in d.columns:
            raise ValueError("Coluna 'ian' necessária para criar o target.")
        d["risco_defasagem"] = (coerce_num(d["ian"]) <= 5).astype(int)

    # Remover colunas de vazamento e não preditivas
    remover = [c for c in COLUNAS_REMOVER + ["ian", "defasagem"] if c in d.columns]
    d = d.drop(columns=remover, errors="ignore")

    return d


# ─── Pipeline de pré-processamento sklearn ────────────────────────────────────

def construir_preprocessador(X_train: pd.DataFrame) -> ColumnTransformer:
    """Constrói ColumnTransformer com imputação mediana + scaling para numérico
    e imputação modo + OHE para categórico."""
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


# ─── Helpers para o app ───────────────────────────────────────────────────────

def preparar_entrada_app(dados: dict, colunas_treino: list) -> pd.DataFrame:
    """
    Recebe um dicionário com os campos preenchidos no app e retorna
    um DataFrame com exatamente as colunas que o modelo espera.
    Colunas ausentes são preenchidas com NaN.
    """
    d = pd.DataFrame([dados])
    d = limpar_base(d)
    d = criar_features_derivadas(d)

    # Garantir que todas as colunas de treino existam (NaN se ausentes)
    for col in colunas_treino:
        if col not in d.columns:
            d[col] = np.nan

    return d[colunas_treino]


def classificar_risco(prob: float, threshold: float) -> dict:
    """Retorna label, cor e recomendação com base na probabilidade."""
    if prob >= threshold:
        return {
            "label": "⚠️ ALTO RISCO DE DEFASAGEM",
            "cor": "error",
            "recomendacao": (
                "Recomenda-se intervenção pedagógica e acompanhamento "
                "psicossocial intensificados. Reavaliar em 30 dias."
            ),
        }
    elif prob >= threshold * 0.7:
        return {
            "label": "🟡 RISCO MODERADO",
            "cor": "warning",
            "recomendacao": (
                "Monitorar de perto os indicadores de engajamento e desempenho. "
                "Considerar reforço preventivo."
            ),
        }
    else:
        return {
            "label": "✅ BAIXO RISCO",
            "cor": "success",
            "recomendacao": (
                "Aluno apresenta perfil dentro do esperado. "
                "Manter acompanhamento padrão."
            ),
        }
