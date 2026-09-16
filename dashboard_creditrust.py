"""
Dashboard CrediTrust Finance — Moteur de Crédit Scoring
Projet Simplon Lyon — Data Analyst — Activité 3/4 (Machine Learning & Classification)
Équipe : Laurie (Régression Logistique), David (Arbre de Décision), Zohair (Random Forest)
Auteur du dashboard : Zohair

Lancement : streamlit run dashboard_creditrust.py
Prérequis : streamlit, pandas, numpy, scikit-learn, plotly
Le fichier loan_data.csv doit être placé dans le même dossier (ou uploadé via la barre latérale).
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# ----------------------------------------------------------------------------
# 1. CONFIGURATION GÉNÉRALE ET PALETTE DE COULEURS
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="CrediTrust Finance — Crédit Scoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#1B3B6F",       # bleu marine — identité CrediTrust
    "primary_light": "#3D5A80",
    "accent": "#2EC4B6",        # teal — accent principal
    "success": "#43AA8B",       # vert-teal — classe Y / prêt accordé
    "danger": "#E63946",        # rouge — classe N / risque
    "warning": "#F4A261",       # ambre — mise en avant / modèle balanced
    "bg": "#F7F9FC",
    "card": "#FFFFFF",
    "text": "#1B2A4A",
    "muted": "#6C7A93",
}

MODEL_COLORS = {
    "Régression Logistique": COLORS["primary_light"],
    "Arbre de Décision": COLORS["warning"],
    "Random Forest": COLORS["accent"],
    "Random Forest (balanced)": COLORS["success"],
}

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {COLORS['bg']};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {COLORS['primary']};
    }}
    section[data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
    }}
    h1, h2, h3 {{
        color: {COLORS['text']};
        font-family: "Segoe UI", sans-serif;
    }}
    .block-container {{
        padding-top: 2rem;
    }}
    div[data-testid="stMetric"] {{
        background-color: {COLORS['card']};
        border: 1px solid #E5E9F0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    .badge-accepte {{
        background-color: {COLORS['success']};
        color: white; padding: 0.6rem 1.2rem; border-radius: 8px;
        font-weight: 600; font-size: 1.1rem; display: inline-block;
    }}
    .badge-refuse {{
        background-color: {COLORS['danger']};
        color: white; padding: 0.6rem 1.2rem; border-radius: 8px;
        font-weight: 600; font-size: 1.1rem; display: inline-block;
    }}
    .info-card {{
        background-color: {COLORS['card']};
        border-left: 5px solid {COLORS['accent']};
        border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1rem;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color=COLORS["text"], family="Segoe UI, sans-serif"),
    margin=dict(l=40, r=20, t=60, b=40),
)


def style_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ----------------------------------------------------------------------------
# 2. CHARGEMENT DES DONNÉES
# ----------------------------------------------------------------------------

CAT_MISSING_COLS = ["Gender", "Married", "Dependents", "Self_Employed"]
NUM_MISSING_COLS = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]
ENCODE_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]


@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df


# ----------------------------------------------------------------------------
# 3. PRÉTRAITEMENT + ENTRAÎNEMENT DES 4 MODÈLES (identique au notebook)
# ----------------------------------------------------------------------------

@st.cache_resource
def train_all_models(df):
    df_train_valid = df[df["Loan_Status"].notna()].copy()

    X = df_train_valid.drop(columns=["Loan_Status", "Loan_ID"])
    y = df_train_valid["Loan_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -- Valeurs manquantes catégorielles (Laurie) --
    for col in CAT_MISSING_COLS:
        X_train[col] = X_train[col].fillna("Inconnu")
        X_test[col] = X_test[col].fillna("Inconnu")

    # -- Valeurs manquantes numériques, médiane calculée sur train (David) --
    medianes = {}
    for col in NUM_MISSING_COLS:
        mediane = X_train[col].median()
        medianes[col] = mediane
        X_train[col] = X_train[col].fillna(mediane)
        X_test[col] = X_test[col].fillna(mediane)

    # -- Encodage catégoriel + alignement des colonnes (Zohair) --
    X_train_enc = pd.get_dummies(X_train, columns=ENCODE_COLS, drop_first=True)
    X_test_enc = pd.get_dummies(X_test, columns=ENCODE_COLS, drop_first=True)
    X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)
    X_test_enc = X_test_enc[X_train_enc.columns]

    # -- Standardisation pour la Régression Logistique --
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    # -- Modèle 1 : Régression Logistique (Laurie) --
    model_1 = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    model_1.fit(X_train_scaled, y_train)
    y_pred_model_1 = model_1.predict(X_test_scaled)

    # -- Modèle 2 : Arbre de Décision (David) --
    model_dtc = DecisionTreeClassifier(
        random_state=0, max_depth=None, min_samples_split=2, criterion="gini"
    )
    model_dtc.fit(X_train_enc, y_train)
    prediction_dtc = model_dtc.predict(X_test_enc)

    # -- Modèle 3 : Random Forest standard (Zohair) --
    model_rf = RandomForestClassifier(random_state=42)
    model_rf.fit(X_train_enc, y_train)
    y_pred_rf = model_rf.predict(X_test_enc)

    # -- Modèle 4 : Random Forest balanced — MODÈLE FINAL RETENU (Zohair) --
    model_rf_balanced = RandomForestClassifier(random_state=42, class_weight="balanced")
    model_rf_balanced.fit(X_train_enc, y_train)
    y_pred_rf_balanced = model_rf_balanced.predict(X_test_enc)

    # -- Tableau comparatif --
    predictions = {
        "Régression Logistique": y_pred_model_1,
        "Arbre de Décision": prediction_dtc,
        "Random Forest": y_pred_rf,
        "Random Forest (balanced)": y_pred_rf_balanced,
    }
    lignes = []
    for nom, y_pred in predictions.items():
        lignes.append({
            "Modèle": nom,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Précision (macro)": precision_score(y_test, y_pred, average="macro"),
            "Rappel (macro)": recall_score(y_test, y_pred, average="macro"),
            "Rappel classe N": recall_score(y_test, y_pred, pos_label="N"),
            "F1 (macro)": f1_score(y_test, y_pred, average="macro"),
        })
    resultats_df = pd.DataFrame(lignes)

    # -- Importance des variables du modèle final --
    importances_df = pd.DataFrame({
        "Variable": X_train_enc.columns,
        "Importance": model_rf_balanced.feature_importances_,
    }).sort_values("Importance", ascending=False)

    # -- Matrice de confusion du modèle final --
    cm = confusion_matrix(y_test, y_pred_rf_balanced, labels=["N", "Y"])

    return {
        "resultats_df": resultats_df,
        "importances_df": importances_df,
        "confusion_matrix": cm,
        "model_final": model_rf_balanced,
        "colonnes_final": X_train_enc.columns,
        "medianes": medianes,
        "n_train": len(X_train_enc),
        "n_test": len(X_test_enc),
    }


def encoder_saisie_utilisateur(saisie: dict, colonnes_reference) -> pd.DataFrame:
    """Applique le même encodage (get_dummies + alignement) qu'à l'entraînement
    à une saisie unique du simulateur, pour la passer au modèle final (RF balanced)."""
    ligne = pd.DataFrame([saisie])
    ligne_enc = pd.get_dummies(ligne, columns=ENCODE_COLS, drop_first=True)
    ligne_enc = ligne_enc.reindex(columns=colonnes_reference, fill_value=0)
    return ligne_enc


# ----------------------------------------------------------------------------
# 4. PAGES DU DASHBOARD
# ----------------------------------------------------------------------------

def page_contexte():
    st.title("💳 CrediTrust Finance — Moteur de Crédit Scoring")
    st.markdown(
        '<div class="info-card">Projet réalisé dans le cadre de la formation '
        '<b>Data Analyst — Simplon Lyon</b>, brief "Machine Learning & Classification" '
        'pour le client fictif <b>CrediTrust Finance</b>.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🎯 Objectif métier")
        st.write(
            "Concevoir un moteur de scoring capable de prédire si une demande de prêt "
            "doit être **accordée (Y)** ou **refusée (N)**, à partir du profil du "
            "demandeur (revenus, historique de crédit, situation familiale, zone "
            "géographique...)."
        )
        st.write(
            "**Priorité métier explicite du brief :** réduire les faux négatifs. "
            "Un mauvais payeur non détecté coûte plus cher à la banque qu'un bon "
            "payeur refusé à tort — le **rappel sur la classe N** (mauvais payeurs) "
            "est donc le critère de choix prioritaire, avant l'accuracy globale."
        )

        st.subheader("👥 Répartition du travail (trinôme)")
        st.markdown(
            """
            - **Laurie** — valeurs manquantes catégorielles (remplacement par "Inconnu") + Régression Logistique
            - **David** — valeurs manquantes numériques (médiane calculée sur train) + Arbre de Décision
            - **Zohair** — encodage des variables catégorielles + alignement train/test + Random Forest (standard et balanced)
            """
        )
    with col2:
        st.subheader("🛠️ Stack technique")
        st.markdown(
            """
            - Python / pandas / scikit-learn
            - CoCalc (notebook partagé temps réel)
            - Streamlit (ce dashboard)
            - 4 modèles comparés, 1 modèle final retenu
            """
        )


def page_eda(df):
    st.title("📊 Exploration des données (EDA)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", df.shape[0])
    c2.metric("Colonnes", df.shape[1])
    c3.metric("Doublons", int(df.duplicated().sum()))
    c4.metric("Valeurs manquantes (total)", int(df.isnull().sum().sum()))

    st.subheader("Répartition de la variable cible")
    repartition = df["Loan_Status"].value_counts(normalize=True).reset_index()
    repartition.columns = ["Loan_Status", "Proportion"]
    fig = px.pie(
        repartition, names="Loan_Status", values="Proportion",
        color="Loan_Status",
        color_discrete_map={"Y": COLORS["success"], "N": COLORS["danger"]},
        hole=0.45,
    )
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Distributions des variables numériques")
    num_col = st.selectbox(
        "Choisir une variable", ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term"]
    )
    fig = px.histogram(
        df, x=num_col, color="Loan_Status", barmode="overlay", nbins=40,
        color_discrete_map={"Y": COLORS["success"], "N": COLORS["danger"]},
    )
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Loan_Status selon les variables catégorielles")
    cat_col = st.selectbox(
        "Choisir une variable catégorielle", ["Credit_History", "Property_Area", "Education", "Married", "Gender"]
    )
    croisement = pd.crosstab(df[cat_col], df["Loan_Status"], normalize="index").reset_index()
    croisement_melt = croisement.melt(id_vars=cat_col, var_name="Loan_Status", value_name="Proportion")
    fig = px.bar(
        croisement_melt, x=cat_col, y="Proportion", color="Loan_Status", barmode="stack",
        color_discrete_map={"Y": COLORS["success"], "N": COLORS["danger"]},
    )
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Corrélation entre variables numériques")
    df_corr_num = df.select_dtypes(include="number").corr()
    fig = px.imshow(
        df_corr_num, text_auto=".2f", color_continuous_scale=["#E63946", "#F7F9FC", "#2EC4B6"],
        aspect="auto",
    )
    st.plotly_chart(style_fig(fig), use_container_width=True)


def page_pretraitement(train_info):
    st.title("🧹 Prétraitement des données")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 1. Séparation train / test")
        st.write("`train_test_split` (80/20, stratifié sur `Loan_Status`, `random_state=42`)")
        st.metric("Lignes train", train_info["n_train"])
        st.metric("Lignes test", train_info["n_test"])

        st.markdown("#### 2. Valeurs manquantes catégorielles")
        st.write(
            f"Colonnes concernées : `{', '.join(CAT_MISSING_COLS)}`. "
            "Remplacement par la catégorie **\"Inconnu\"** plutôt que le mode, "
            "pour ne pas introduire de biais artificiel vers la catégorie majoritaire."
        )

        st.markdown("#### 3. Valeurs manquantes numériques")
        st.write(f"Colonnes concernées : `{', '.join(NUM_MISSING_COLS)}`.")
        st.write("Médiane calculée **sur le train uniquement**, puis appliquée au train et au test (pas de fuite de données) :")
        st.table(pd.DataFrame(train_info["medianes"], index=["Médiane (train)"]).T)

    with col2:
        st.markdown("#### 4. Encodage des variables catégorielles")
        st.write(f"`pd.get_dummies` (`drop_first=True`) sur : `{', '.join(ENCODE_COLS)}`.")
        st.write(
            "Un décalage de colonnes entre train et test (catégories absentes d'un "
            "des deux jeux) a été corrigé avec `.align(join='left', fill_value=0)`."
        )

        st.markdown("#### 5. Standardisation")
        st.write(
            "`StandardScaler` appliqué **uniquement pour la Régression Logistique** "
            "(sensible à l'échelle des variables). Les modèles à base d'arbres "
            "(Arbre de Décision, Random Forest) n'en ont pas besoin."
        )

        st.markdown("#### Récapitulatif du pipeline")
        st.code(
            "Données brutes\n"
            "  → split train/test (stratifié)\n"
            "  → imputation catégorielle ('Inconnu')\n"
            "  → imputation numérique (médiane train)\n"
            "  → get_dummies + align\n"
            "  → [StandardScaler] (LogReg uniquement)\n"
            "  → entraînement des 4 modèles",
            language="text",
        )


def page_comparaison(resultats_df):
    st.title("🤖 Comparaison des modèles")

    st.dataframe(
        resultats_df.style.format({c: "{:.3f}" for c in resultats_df.columns if c != "Modèle"})
        .highlight_max(subset=["Rappel classe N"], color=COLORS["success"] + "40"),
        use_container_width=True,
    )

    metrique = st.radio(
        "Métrique à visualiser", ["Rappel classe N", "Accuracy", "F1 (macro)", "Précision (macro)"],
        horizontal=True,
    )
    fig = px.bar(
        resultats_df, x="Modèle", y=metrique, color="Modèle",
        color_discrete_map=MODEL_COLORS, text_auto=".3f",
    )
    fig.update_layout(showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown(
        '<div class="info-card">'
        '<b>Modèle final retenu : Random Forest (balanced).</b> Le brief priorise la '
        'réduction des faux négatifs (un mauvais payeur non détecté coûte plus cher à '
        'la banque qu\'un bon payeur refusé à tort), donc le rappel sur la classe N est '
        'le critère de choix principal, pas l\'accuracy globale. Avec le meilleur rappel '
        'classe N du comparatif, ce modèle détecte le mieux les mauvais payeurs — au '
        'prix d\'une accuracy légèrement inférieure à la Régression Logistique. '
        'Un compromis assumé, cohérent avec l\'objectif métier.'
        '</div>',
        unsafe_allow_html=True,
    )


def page_importance(importances_df, cm):
    st.title("🌟 Importance des variables — modèle final")

    top_n = st.slider("Nombre de variables à afficher", 5, len(importances_df), 10)
    df_top = importances_df.head(top_n).sort_values("Importance")
    fig = px.bar(
        df_top, x="Importance", y="Variable", orientation="h",
        color="Importance", color_continuous_scale=["#3D5A80", "#2EC4B6", "#43AA8B"],
    )
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.subheader("Matrice de confusion (Random Forest balanced)")
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Prédit N", "Prédit Y"], y=["Réel N", "Réel Y"],
        color_continuous_scale=["#F7F9FC", COLORS["primary"]],
    )
    st.plotly_chart(style_fig(fig_cm), use_container_width=True)


def page_simulateur(model_final, colonnes_final):
    st.title("🧮 Simulateur de prêt")
    st.write("Renseigne un profil pour obtenir une décision instantanée du modèle final (Random Forest balanced).")

    with st.form("simulateur"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Genre", ["Male", "Female"])
            married = st.selectbox("Marié(e)", ["Yes", "No"])
            dependents = st.selectbox("Nombre de personnes à charge", ["0", "1", "2", "3+"])
            education = st.selectbox("Éducation", ["Graduate", "Not Graduate"])
        with col2:
            self_employed = st.selectbox("Indépendant", ["Yes", "No"])
            property_area = st.selectbox("Zone", ["Urban", "Semiurban", "Rural"])
            credit_history = st.radio("Historique de crédit favorable", ["Oui", "Non"], horizontal=True)
        with col3:
            applicant_income = st.number_input("Revenu du demandeur", min_value=0, value=5000, step=100)
            coapplicant_income = st.number_input("Revenu du co-demandeur", min_value=0, value=0, step=100)
            loan_amount = st.number_input("Montant du prêt (en milliers)", min_value=0, value=150, step=10)
            loan_term = st.selectbox("Durée du prêt (mois)", [360, 180, 120, 84, 60, 36, 12], index=0)

        submitted = st.form_submit_button("Évaluer la demande", use_container_width=True)

    if submitted:
        saisie = {
            "Gender": gender, "Married": married, "Dependents": dependents,
            "Education": education, "Self_Employed": self_employed,
            "Property_Area": property_area,
            "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount, "Loan_Amount_Term": loan_term,
            "Credit_History": 1.0 if credit_history == "Oui" else 0.0,
        }
        ligne_enc = encoder_saisie_utilisateur(saisie, colonnes_final)
        prediction = model_final.predict(ligne_enc)[0]
        proba = model_final.predict_proba(ligne_enc)[0]
        classes = list(model_final.classes_)
        proba_y = proba[classes.index("Y")]

        st.divider()
        col1, col2 = st.columns([1, 2])
        with col1:
            if prediction == "Y":
                st.markdown('<span class="badge-accepte">✅ Prêt accordé</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-refuse">❌ Prêt refusé</span>', unsafe_allow_html=True)
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba_y * 100,
                number={"suffix": "%"},
                title={"text": "Probabilité d'accord (classe Y)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": COLORS["primary"]},
                    "steps": [
                        {"range": [0, 50], "color": COLORS["danger"] + "50"},
                        {"range": [50, 100], "color": COLORS["success"] + "50"},
                    ],
                },
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)


def page_recommandations():
    st.title("📝 Recommandations et limites")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Recommandations")
        st.markdown(
            """
            - Déployer le **Random Forest (balanced)** comme moteur de décision, avec revue humaine systématique des cas proches du seuil de décision.
            - Fixer un **seuil de probabilité métier** (pas nécessairement 50/50) pour ajuster davantage le compromis précision/rappel selon l'appétit au risque de la banque.
            - Suivre le modèle dans le temps (drift) : la distribution des profils demandeurs peut évoluer.
            - Compléter par une variable de suivi des remboursements réels pour ré-entraîner périodiquement.
            """
        )
    with col2:
        st.subheader("⚠️ Limites")
        st.markdown(
            """
            - Jeu de données de taille modeste : risque de sur-apprentissage sur certains profils rares.
            - Déséquilibre des classes (~69% Y / 31% N) partiellement corrigé par `class_weight='balanced'`, mais pas éliminé.
            - La catégorie "Inconnu" (valeurs manquantes) peut masquer un signal réel plutôt que du hasard.
            - Le modèle reflète des données historiques : il peut reproduire des biais déjà présents dans les décisions passées (ex. zone géographique, statut marital).
            - Pas de validation externe (autre banque, autre période) à ce stade.
            """
        )


# ----------------------------------------------------------------------------
# 5. NAVIGATION
# ----------------------------------------------------------------------------

def main():
    st.sidebar.title("💳 CrediTrust")
    st.sidebar.caption("Simplon Lyon — Data Analyst")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🏠 Contexte",
            "📊 Exploration des données",
            "🧹 Prétraitement",
            "🤖 Comparaison des modèles",
            "🌟 Importance des variables",
            "🧮 Simulateur de prêt",
            "📝 Recommandations & limites",
        ],
    )

    st.sidebar.divider()
    uploaded = st.sidebar.file_uploader("loan_data.csv (si absent du dossier)", type="csv")

    try:
        df = load_data(uploaded if uploaded is not None else "loan_data.csv")
    except FileNotFoundError:
        st.error(
            "Fichier `loan_data.csv` introuvable. Place-le dans le même dossier que "
            "ce script, ou dépose-le via la barre latérale."
        )
        st.stop()

    if page == "🏠 Contexte":
        page_contexte()
        return
    if page == "📊 Exploration des données":
        page_eda(df)
        return

    with st.spinner("Entraînement des modèles..."):
        train_info = train_all_models(df)

    if page == "🧹 Prétraitement":
        page_pretraitement(train_info)
    elif page == "🤖 Comparaison des modèles":
        page_comparaison(train_info["resultats_df"])
    elif page == "🌟 Importance des variables":
        page_importance(train_info["importances_df"], train_info["confusion_matrix"])
    elif page == "🧮 Simulateur de prêt":
        page_simulateur(train_info["model_final"], train_info["colonnes_final"])
    elif page == "📝 Recommandations & limites":
        page_recommandations()


if __name__ == "__main__":
    main()
