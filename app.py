"""
app.py — Passos Mágicos: Preditor de Risco de Defasagem Escolar
Streamlit Community Cloud deploy
"""
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path

# ─── Importar utilitários de pré-processamento ────────────────────────────────
from utils_pm import preparar_entrada_app, classificar_risco, limpar_base, criar_features_derivadas

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Passos Mágicos — Risco de Defasagem",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS customizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2e86c1 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .risk-card-high   { background: #fdecea; border-left: 5px solid #e53935; padding: 1rem; border-radius: 8px; }
    .risk-card-medium { background: #fff8e1; border-left: 5px solid #fb8c00; padding: 1rem; border-radius: 8px; }
    .risk-card-low    { background: #e8f5e9; border-left: 5px solid #43a047; padding: 1rem; border-radius: 8px; }
    .metric-box { text-align: center; padding: 0.7rem; border-radius: 8px; background: #f0f4f8; margin: 0.3rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── Carregar artefatos (cacheado) ────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo…")
def carregar_modelo():
    base = Path(__file__).parent
    model_path = base / "modelo" / "modelo_passos_magicos.pkl"
    cfg_path   = base / "modelo" / "config_passos_magicos.pkl"
    feat_path  = base / "modelo" / "feature_names.pkl"

    if not model_path.exists():
        return None, None, None

    modelo    = joblib.load(model_path)
    config    = joblib.load(cfg_path)
    feat_names = joblib.load(feat_path) if feat_path.exists() else None
    return modelo, config, feat_names


modelo, config, feat_names = carregar_modelo()
THRESHOLD = float(config["threshold"]) if config else 0.5
MODEL_NAME = config.get("best_model", "Modelo") if config else "Modelo"


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎓 Passos Mágicos — Preditor de Risco de Defasagem</h1>
    <p style='margin:0; opacity:.85'>Ferramenta de apoio pedagógico baseada em Machine Learning para identificação
    antecipada de alunos em risco de defasagem escolar.</p>
</div>
""", unsafe_allow_html=True)

if modelo is None:
    st.error(
        "⚠️ Modelo não encontrado. Execute o notebook de treinamento primeiro para gerar "
        "os arquivos na pasta `modelo/`."
    )
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/school.png", width=60)
    st.title("Navegação")
    aba = st.radio("Selecione a funcionalidade:", ["📋 Predição Individual", "📂 Análise em Lote"])
    st.divider()
    st.markdown(f"**Modelo ativo:** `{MODEL_NAME}`")
    st.markdown(f"**Threshold:** `{THRESHOLD:.2f}`")
    st.divider()
    st.caption("Passos Mágicos · FIAP Datathon 2026")


# ══════════════════ PREDIÇÃO INDIVIDUAL ═══════════════════════════════════════
if aba == "📋 Predição Individual":
    st.subheader("📋 Análise Individual do Aluno")
    st.info("Preencha os indicadores do aluno e clique em **Analisar Risco**.")

    with st.form("form_individual"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**👤 Dados do Aluno**")
            genero     = st.selectbox("Gênero", ["masculino", "feminino"])
            idade      = st.slider("Idade", 6, 30, 14)
            fase_ideal = st.number_input("Fase Ideal", 0.0, 8.0, 3.0, step=1.0)

        with c2:
            st.markdown("**📚 Notas Acadêmicas (0–10)**")
            mat = st.slider("Matemática (MAT)", 0.0, 10.0, 6.0, step=0.1)
            por = st.slider("Português (POR)",  0.0, 10.0, 6.0, step=0.1)
            ing = st.slider("Inglês (ING)",      0.0, 10.0, 6.0, step=0.1)

        with c3:
            st.markdown("**📊 Indicadores (0–10)**")
            iaa = st.slider("Ind. Autoavaliação (IAA)", 0.0, 10.0, 6.0, step=0.1)
            ieg = st.slider("Ind. Engajamento (IEG)",   0.0, 10.0, 6.0, step=0.1)
            ips = st.slider("Ind. Psicossocial (IPS)",  0.0, 10.0, 6.0, step=0.1)
            ipp = st.slider("Ind. Psicopedagógico (IPP)", 0.0, 10.0, 6.0, step=0.1)

        st.markdown("---")
        c4, c5 = st.columns(2)
        with c4:
            st.markdown("**📈 Histórico INDE**")
            inde_2022 = st.number_input("INDE 2022", 0.0, 10.0, 6.0, step=0.01)
            inde_2023 = st.number_input("INDE 2023", 0.0, 10.0, 6.5, step=0.01)
        with c5:
            st.markdown("**🔍 Outros Indicadores**")
            ida = st.number_input("IDA (Desempenho Acad.)",    0.0, 10.0, 6.0, step=0.01)
            ipv = st.number_input("IPV (Ponto de Virada)",     0.0, 10.0, 5.0, step=0.01)
            n_av = st.number_input("Nº de Avaliações", 1, 10, 3, step=1)

        submitted = st.form_submit_button("🔍 Analisar Risco", type="primary", use_container_width=True)

    if submitted:
        dados = {
            "genero": genero, "idade": float(idade), "fase_ideal": float(fase_ideal),
            "mat": mat, "por": por, "ing": ing,
            "iaa": iaa, "ieg": ieg, "ips": ips, "ipp": ipp,
            "inde_2022": inde_2022, "inde_2023": inde_2023,
            "ida": ida, "ipv": ipv, "n_av": float(n_av),
        }

        colunas = feat_names if feat_names else list(dados.keys())
        X_input = preparar_entrada_app(dados, colunas)

        with st.spinner("Calculando probabilidade de risco…"):
            prob = float(modelo.predict_proba(X_input)[:, 1][0])

        resultado = classificar_risco(prob, THRESHOLD)
        cor = resultado["cor"]
        label = resultado["label"]
        rec   = resultado["recomendacao"]

        st.markdown("---")
        st.subheader("📊 Resultado da Análise")

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            css_class = {"error": "risk-card-high", "warning": "risk-card-medium",
                         "success": "risk-card-low"}.get(cor, "risk-card-low")
            st.markdown(f"""
            <div class="{css_class}">
                <h3>{label}</h3>
                <p><b>Probabilidade de risco:</b> {prob*100:.1f}%</p>
                <p><b>Threshold do modelo:</b> {THRESHOLD:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_r2:
            if cor == "error":
                st.error(f"**Recomendação:** {rec}")
            elif cor == "warning":
                st.warning(f"**Recomendação:** {rec}")
            else:
                st.success(f"**Recomendação:** {rec}")

            # Indicadores que mais chamam atenção
            indicadores_baixos = {k: v for k, v in dados.items()
                                  if k in ("iaa","ieg","ips","ipp","ida","ipv","mat","por","ing")
                                  and isinstance(v, float) and v < 5.0}
            if indicadores_baixos:
                st.markdown("**⚠️ Indicadores abaixo de 5.0:**")
                for k, v in sorted(indicadores_baixos.items(), key=lambda x: x[1]):
                    st.markdown(f"  - `{k.upper()}` = {v:.1f}")

        # Barra de progresso
        st.markdown("**Probabilidade de Risco:**")
        color_bar = "🔴" if prob >= THRESHOLD else ("🟠" if prob >= THRESHOLD*0.7 else "🟢")
        st.progress(min(prob, 1.0))
        st.caption(f"{color_bar} {prob*100:.1f}% de probabilidade de defasagem")


# ══════════════════ ANÁLISE EM LOTE ═══════════════════════════════════════════
else:
    st.subheader("📂 Análise em Lote")
    st.info(
        "Faça upload de um arquivo **Excel (.xlsx)** ou **CSV** com os dados dos alunos. "
        "O arquivo deve conter as colunas com os indicadores (genero, idade, mat, por, ing, iaa, ieg, ips, ipp, etc.)."
    )

    uploaded = st.file_uploader("Selecione o arquivo", type=["xlsx", "csv"])

    if uploaded:
        try:
            if uploaded.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded)
            else:
                df_up = pd.read_excel(uploaded)

            st.success(f"✅ Arquivo carregado: **{uploaded.name}** — {len(df_up)} registros")
            st.dataframe(df_up.head())

            if st.button("🚀 Processar Todos os Alunos", type="primary"):
                with st.spinner("Processando…"):
                    df_proc = limpar_base(df_up)
                    df_proc = criar_features_derivadas(df_proc)

                    # Alinha colunas com o modelo
                    if feat_names:
                        for col in feat_names:
                            if col not in df_proc.columns:
                                df_proc[col] = np.nan
                        X_batch = df_proc[feat_names]
                    else:
                        X_batch = df_proc.select_dtypes(include=[np.number])

                    probs = modelo.predict_proba(X_batch)[:, 1]
                    preds = (probs >= THRESHOLD).astype(int)

                df_resultado = df_up.copy()
                df_resultado["Probabilidade (%)"] = (probs * 100).round(1)
                df_resultado["Threshold"]          = round(THRESHOLD, 2)
                df_resultado["Classificação"]      = [
                    "⚠️ ALTO RISCO" if p else "✅ BAIXO RISCO" for p in preds
                ]
                df_resultado["Recomendação"] = [
                    "Intervenção intensificada" if p else "Acompanhamento padrão" for p in preds
                ]

                st.subheader("📋 Resultados")
                n_risco = int(preds.sum())
                st.metric("Alunos em ALTO RISCO", n_risco,
                          delta=f"{n_risco/len(preds)*100:.1f}% do total")

                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.dataframe(
                        df_resultado[["Probabilidade (%)", "Classificação", "Recomendação"]],
                        use_container_width=True
                    )
                with col_e2:
                    st.bar_chart(df_resultado["Classificação"].value_counts())

                # Download
                import io
                buf = io.BytesIO()
                df_resultado.to_excel(buf, index=False)
                st.download_button(
                    "⬇️ Baixar Resultados (.xlsx)",
                    data=buf.getvalue(),
                    file_name="resultado_risco_defasagem.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

    else:
        # Modelo de arquivo para download
        exemplo = pd.DataFrame([{
            "genero": "masculino", "idade": 14, "fase_ideal": 3,
            "mat": 5.5, "por": 6.0, "ing": 4.5,
            "iaa": 6.5, "ieg": 5.0, "ips": 6.0, "ipp": 5.5,
            "inde_2022": 5.8, "inde_2023": 6.2, "ida": 5.5, "ipv": 4.0, "n_av": 3,
        }])
        import io
        buf = io.BytesIO()
        exemplo.to_excel(buf, index=False)
        st.download_button(
            "⬇️ Baixar Modelo de Planilha",
            data=buf.getvalue(),
            file_name="modelo_entrada_alunos.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; font-size:0.85em;'>
    🎓 <b>Passos Mágicos</b> · Ferramenta Preditiva de Defasagem Escolar<br>
    Desenvolvido como entrega do Datathon FIAP Fase 5 — Pós-Graduação em Data Analytics<br>
    ⚠️ Este sistema é uma ferramenta de apoio e não substitui a avaliação pedagógica profissional.
</div>
""", unsafe_allow_html=True)
