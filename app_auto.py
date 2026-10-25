
"""
app.py
================================================================================
Interface Streamlit : EAC-KMeans avec G-Split Agent Auto-K
================================================================================

Objectif de l'interface
-----------------------
L'utilisateur ne choisit que le dataset CSV.

Tout le reste est automatique :
    - prétraitement numérique,
    - standardisation,
    - choix automatique de k,
    - optimisation par agents,
    - calcul des gains,
    - affichage de l'efficacité.

Cette interface est volontairement centrée sur la justification du choix de k.
Elle ne dépend pas de l'ACP/PCA pour expliquer l'approche.
"""

import glob
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from my_kmeans_auto import EACKMeans


# =============================================================================
# CONFIGURATION DE PAGE
# =============================================================================

st.set_page_config(
    page_title="G-Split Agent Auto-K",
    layout="wide",
)

st.title("EAC-KMeans : choix automatique de k par G-Split Agent Auto-K")

st.markdown("""
Cette application est volontairement automatique : **l'utilisateur choisit seulement le dataset**.

Le but principal est de montrer comment notre approche choisit automatiquement `k`,
puis optimise les clusters avec une logique d'agents.
""")


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def load_csv(path):
    """
    Charge un fichier CSV avec pandas.
    """
    return pd.read_csv(path)


def select_numeric_data(df):
    """
    Sélectionne uniquement les colonnes numériques.

    K-means travaille avec des distances euclidiennes.
    Ces distances n'ont pas de sens direct sur des textes ou des catégories.
    """
    return df.select_dtypes(include=[np.number]).copy()


def preprocess_data(df_numeric):
    """
    Prétraitement entièrement automatique.

    Étape 1 :
        Remplacer les valeurs manquantes par la moyenne de chaque colonne.

    Étape 2 :
        Standardiser les colonnes.
        Ainsi, toutes les variables participent équitablement aux distances.
    """
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(df_numeric)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X


def safe_internal_scores(X, labels):
    """
    Calcule des scores de qualité du clustering.

    Attention :
        Les outliers ont le label -1.
        Pour les scores internes, on ignore les outliers.

    Scores :
        Silhouette :
            plus grand = mieux.

        Davies-Bouldin :
            plus petit = mieux.

        Calinski-Harabasz :
            plus grand = mieux.
    """
    labels = np.asarray(labels)
    mask = labels != -1

    X_valid = X[mask]
    labels_valid = labels[mask]

    if len(np.unique(labels_valid)) < 2 or len(X_valid) <= len(np.unique(labels_valid)):
        return {
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }

    try:
        sil = silhouette_score(X_valid, labels_valid)
    except Exception:
        sil = np.nan

    try:
        db = davies_bouldin_score(X_valid, labels_valid)
    except Exception:
        db = np.nan

    try:
        ch = calinski_harabasz_score(X_valid, labels_valid)
    except Exception:
        ch = np.nan

    return {
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
    }


def decisions_to_dataframe(history):
    """
    Transforme l'historique des décisions G-Split en tableau lisible.
    """
    rows = []

    for d in history:
        rows.append({
            "Tour": d["round"],
            "Cluster-agent": d["cluster"],
            "Taille cluster": d["size"],
            "Décision": "Acceptée" if d["accepted"] else "Refusée",
            "Raison": d["reason"],
            "Inertie avant": d["inertia_before"],
            "Inertie après": d["inertia_after"],
            "Gain inertie": d["inertia_gain"],
            "Silhouette locale": d["silhouette_local"],
            "Signal G-Means": d["gaussian_signal"],
            "Taille sous-cluster 1": d["subcluster_1_size"],
            "Taille sous-cluster 2": d["subcluster_2_size"],
        })

    return pd.DataFrame(rows)


def run_classic_kmeans(X, k):
    """
    Lance un K-means classique avec le même k que notre méthode.
    Cette comparaison sert à mesurer le gain de notre optimisation par agents.
    """
    start = time.time()

    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42,
    )

    labels = km.fit_predict(X)
    elapsed = time.time() - start

    return km, labels, elapsed


def plot_k_evolution(decisions_df):
    """
    Trace la construction progressive de k.

    k commence à 2.
    Chaque division acceptée augmente k de 1.
    """
    accepted = decisions_df[decisions_df["Décision"] == "Acceptée"]

    k_values = [2]
    steps = [0]

    k = 2

    for i in range(len(accepted)):
        k += 1
        k_values.append(k)
        steps.append(i + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(steps, k_values, marker="o")
    ax.set_title("Construction automatique de k")
    ax.set_xlabel("Nombre de divisions acceptées")
    ax.set_ylabel("k courant")
    ax.grid(True, alpha=0.3)

    return fig


def plot_acceptance_summary(decisions_df):
    """
    Affiche combien de décisions ont été acceptées et refusées.
    """
    counts = decisions_df["Décision"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title("Décisions de division des agents")
    ax.set_xlabel("Type de décision")
    ax.set_ylabel("Nombre")
    ax.grid(True, axis="y", alpha=0.3)

    return fig


def plot_split_gains(decisions_df):
    """
    Affiche les gains d'inertie des divisions testées.

    Les divisions acceptées doivent avoir un gain suffisant.
    """
    df = decisions_df.dropna(subset=["Gain inertie"]).copy()

    if df.empty:
        return None

    df["Nom"] = df.apply(lambda r: f"T{int(r['Tour'])}-C{int(r['Cluster-agent'])}", axis=1)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(df["Nom"], df["Gain inertie"])
    ax.axhline(0.20, linestyle="--", label="Seuil automatique 20%")
    ax.set_title("Gain d'inertie locale des divisions testées")
    ax.set_xlabel("Division testée")
    ax.set_ylabel("Gain d'inertie")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_quality_criteria(decisions_df):
    """
    Affiche les critères de qualité des divisions :
        - silhouette locale,
        - signal G-Means.
    """
    df = decisions_df.dropna(subset=["Silhouette locale", "Signal G-Means"]).copy()

    if df.empty:
        return None

    df["Nom"] = df.apply(lambda r: f"T{int(r['Tour'])}-C{int(r['Cluster-agent'])}", axis=1)

    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(x, df["Silhouette locale"], marker="o", label="Silhouette locale")
    ax.plot(x, df["Signal G-Means"], marker="o", label="Signal G-Means")

    ax.axhline(0.35, linestyle="--", label="Seuil silhouette 0.35")
    ax.axhline(1.00, linestyle=":", label="Seuil G-Means 1.00")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Nom"], rotation=45, ha="right")
    ax.set_title("Critères de qualité pour décider la division")
    ax.set_ylabel("Valeur")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_convergence(model):
    """
    Affiche la convergence de l'inertie globale.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    y = model.history_["inertia"]
    x = np.arange(1, len(y) + 1)

    ax.plot(x, y, marker="o")
    ax.set_title("Optimisation par agents : évolution de l'inertie")
    ax.set_xlabel("Itération")
    ax.set_ylabel("Inertie globale")
    ax.grid(True, alpha=0.3)

    return fig


def plot_transfers(model):
    """
    Affiche les transferts de points faibles entre agents.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    y = model.history_["transfers"]
    x = np.arange(1, len(y) + 1)

    ax.bar(x, y)
    ax.set_title("Transferts de points faibles entre agents")
    ax.set_xlabel("Itération")
    ax.set_ylabel("Nombre de transferts")
    ax.grid(True, axis="y", alpha=0.3)

    return fig


def plot_simple_2d(df_numeric, labels):
    """
    Visualisation simple sans ACP.

    On utilise seulement les deux premières colonnes numériques originales.
    Cette figure n'intervient pas dans le choix de k.
    Elle sert uniquement à donner un aperçu visuel.
    """
    if df_numeric.shape[1] < 2:
        return None

    x_col = df_numeric.columns[0]
    y_col = df_numeric.columns[1]

    fig, ax = plt.subplots(figsize=(8, 5))

    for lab in sorted(np.unique(labels)):
        mask = labels == lab

        if lab == -1:
            ax.scatter(df_numeric.loc[mask, x_col], df_numeric.loc[mask, y_col],
                       marker="x", s=70, label="Outliers")
        else:
            ax.scatter(df_numeric.loc[mask, x_col], df_numeric.loc[mask, y_col],
                       s=35, alpha=0.75, label=f"Cluster {lab}")

    ax.set_title("Aperçu avec deux variables originales, sans ACP")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


# =============================================================================
# BARRE LATERALE : UNIQUEMENT LE CHOIX DU DATASET
# =============================================================================

with st.sidebar:
    st.header("Choix du dataset")

    csv_files = sorted(glob.glob("*.csv"))

    if len(csv_files) == 0:
        st.error("Aucun fichier CSV trouvé dans le dossier courant.")
        st.stop()

    selected_file = st.selectbox(
        "Sélectionner un fichier CSV",
        options=csv_files,
    )

    run_button = st.button(
        "Lancer l'analyse automatique",
        type="primary",
        use_container_width=True,
    )

    st.divider()

    st.info("""
    Aucun paramètre algorithmique n'est demandé à l'utilisateur.

    L'algorithme règle automatiquement :
    - le choix de k ;
    - les seuils de division ;
    - les transferts entre agents ;
    - la détection des outliers.
    """)


# =============================================================================
# CHARGEMENT DU DATASET
# =============================================================================

df = load_csv(selected_file)
numeric_df = select_numeric_data(df)

st.subheader("Dataset sélectionné")

c1, c2, c3 = st.columns(3)
c1.metric("Fichier", selected_file)
c2.metric("Nombre de lignes", df.shape[0])
c3.metric("Colonnes numériques utilisées", numeric_df.shape[1])

st.dataframe(df.head(10), use_container_width=True)

if numeric_df.shape[1] < 1:
    st.error("Le dataset doit contenir au moins une colonne numérique.")
    st.stop()


# =============================================================================
# EXECUTION AUTOMATIQUE
# =============================================================================

if run_button:

    X = preprocess_data(numeric_df)

    with st.spinner("Exécution automatique : choix de k + optimisation par agents..."):
        model = EACKMeans()
        model.fit(X)

    best_k = model._actual_k_
    report = model.optimization_report_
    decisions_df = decisions_to_dataframe(model.gsplit_info_["history"])

    classic_kmeans, classic_labels, classic_time = run_classic_kmeans(X, best_k)

    scores_ours = safe_internal_scores(X, model.final_labels_)
    scores_classic = safe_internal_scores(X, classic_labels)

    # -------------------------------------------------------------------------
    # RESULTATS PRINCIPAUX
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("Résultat principal")

    r1, r2, r3, r4 = st.columns(4)

    r1.metric("k choisi automatiquement", best_k)
    r2.metric("Divisions acceptées", report["accepted_splits"])
    r3.metric("Gain d'inertie", f"{report['inertia_gain_percent']:.2f}%")
    r4.metric("Transferts entre agents", report["total_transfers"])

    st.success(
        f"Le modèle a choisi automatiquement k = {best_k}. "
        "Ce choix provient des divisions acceptées par les clusters-agents, "
        "et non d'un réglage manuel de l'utilisateur."
    )

    # -------------------------------------------------------------------------
    # GAINS ET EFFICACITE
    # -------------------------------------------------------------------------
    st.subheader("Gains, efficacité et optimisation")

    g1, g2, g3, g4 = st.columns(4)

    g1.metric(
        "Inertie K-means classique",
        f"{report['baseline_inertia_same_k']:.4f}",
    )

    g2.metric(
        "Inertie finale EAC",
        f"{report['final_inertia']:.4f}",
    )

    g3.metric(
        "Gain absolu d'inertie",
        f"{report['inertia_gain']:.4f}",
    )

    g4.metric(
        "Temps total",
        f"{report['total_time']:.3f} s",
    )

    efficiency_df = pd.DataFrame({
        "Méthode": ["K-means classique", "EAC-KMeans G-Split"],
        "k": [best_k, best_k],
        "Inertie": [classic_kmeans.inertia_, report["final_inertia"]],
        "Silhouette": [scores_classic["silhouette"], scores_ours["silhouette"]],
        "Davies-Bouldin": [scores_classic["davies_bouldin"], scores_ours["davies_bouldin"]],
        "Calinski-Harabasz": [scores_classic["calinski_harabasz"], scores_ours["calinski_harabasz"]],
        "Temps exécution": [classic_time, report["total_time"]],
        "Itérations": [classic_kmeans.n_iter_, report["iterations"]],
        "Outliers": [0, report["outliers"]],
    })

    st.dataframe(efficiency_df, use_container_width=True)

    if report["inertia_gain"] > 0:
        st.success(
            f"Notre approche réduit l'inertie de {report['inertia_gain_percent']:.2f}% "
            "par rapport à un K-means classique utilisant le même k."
        )
    else:
        st.info(
            "Sur ce dataset, le gain d'inertie est faible ou négatif. "
            "L'intérêt principal reste le choix automatique et justifié de k."
        )

    # -------------------------------------------------------------------------
    # EXPLICATION AUTOMATIQUE
    # -------------------------------------------------------------------------
    st.subheader("Explication automatique du choix de k")

    st.markdown(f"""
    L'algorithme commence avec `k = 2`.

    Ensuite, chaque cluster-agent teste s'il peut se diviser en deux.
    Une division est acceptée uniquement si elle respecte les conditions suivantes :

    1. le cluster initial est assez grand ;
    2. les deux sous-clusters sont assez grands ;
    3. le gain d'inertie locale est au moins **20%** ;
    4. la silhouette locale est au moins **0.35** ;
    5. le signal inspiré de G-Means est au moins **1.00**.

    Dans cette exécution :
    - nombre de divisions acceptées : **{report["accepted_splits"]}** ;
    - nombre final de clusters : **k = {best_k}** ;
    - temps de sélection de k : **{report["selection_time"]:.3f} s**.
    """)

    # -------------------------------------------------------------------------
    # ONGLETS D'ANALYSE
    # -------------------------------------------------------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Décisions de k",
        "Graphiques Auto-K",
        "Optimisation agents",
        "Visualisation simple",
        "Export"
    ])

    with tab1:
        st.markdown("### Historique des décisions de division")
        st.dataframe(decisions_df, use_container_width=True)

        st.markdown("""
        Chaque ligne correspond à une décision prise par un cluster-agent.
        Si la décision est acceptée, alors un nouveau cluster est créé et k augmente de 1.
        """)

    with tab2:
        st.markdown("### Construction progressive de k")
        st.pyplot(plot_k_evolution(decisions_df))

        st.markdown("### Résumé des décisions")
        st.pyplot(plot_acceptance_summary(decisions_df))

        fig_gain = plot_split_gains(decisions_df)
        if fig_gain is not None:
            st.pyplot(fig_gain)

        fig_quality = plot_quality_criteria(decisions_df)
        if fig_quality is not None:
            st.pyplot(fig_quality)

    with tab3:
        st.markdown("### Optimisation par agents après le choix de k")
        st.pyplot(plot_convergence(model))
        st.pyplot(plot_transfers(model))

        st.markdown(f"""
        Après la sélection automatique de `k`, les agents continuent à optimiser les clusters.

        Résumé :
        - transferts totaux : **{report["total_transfers"]}** ;
        - outliers détectés : **{report["outliers"]}** ;
        - itérations : **{report["iterations"]}** ;
        - inertie finale : **{report["final_inertia"]:.4f}**.
        """)

    with tab4:
        st.markdown("### Visualisation simple sans ACP")

        fig = plot_simple_2d(numeric_df.reset_index(drop=True), model.final_labels_)

        if fig is not None:
            st.pyplot(fig)
        else:
            st.info("Il faut au moins deux colonnes numériques pour afficher ce graphique.")

        st.caption(
            "Cette visualisation utilise les deux premières colonnes numériques originales. "
            "Elle ne participe pas au choix de k."
        )

    with tab5:
        st.markdown("### Dataset annoté")

        result_df = df.copy()
        result_df["cluster_eac_gsplit"] = model.final_labels_
        result_df["is_outlier"] = model.outlier_mask_.astype(int)

        st.dataframe(result_df, use_container_width=True)

        csv_export = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Télécharger le dataset annoté",
            data=csv_export,
            file_name="resultats_eac_gsplit.csv",
            mime="text/csv",
        )

else:
    st.info("Choisis seulement le dataset dans la barre latérale, puis lance l'analyse automatique.")
