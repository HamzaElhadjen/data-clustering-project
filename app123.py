
"""
app.py
Interface Streamlit pour EAC-KMeans avec G-Split Agent Auto-K.

Important :
- Cette interface ne repose pas sur l'ACP/PCA.
- Elle se concentre sur la justification du choix automatique de k.
- Les graphiques montrent :
    1) les décisions de division des agents,
    2) le k final,
    3) le gain d'inertie local,
    4) la silhouette locale,
    5) la convergence de l'algorithme.
"""

import glob
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from my_kmeans import EACKMeans


# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="G-Split Agent Auto-K",
    layout="wide",
)

st.title("EAC-KMeans avec choix automatique de k par G-Split Agent Auto-K")

st.markdown("""
Cette interface montre surtout **comment le nombre de clusters k est choisi automatiquement**.

Contrairement à une recherche exhaustive qui teste plusieurs valeurs de `k`, cette approche commence avec un petit nombre de clusters, puis laisse chaque **cluster-agent** décider s'il doit se diviser.
Une division est acceptée seulement si elle est justifiée par des critères mesurables.
""")


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def load_csv(path):
    """Charge un fichier CSV."""
    return pd.read_csv(path)


def select_numeric_data(df):
    """
    Garde uniquement les colonnes numériques.

    Les distances de K-means sont calculées sur des variables numériques.
    Les colonnes textuelles ne sont donc pas utilisées ici.
    """
    return df.select_dtypes(include=[np.number]).copy()


def preprocess_numeric_data(df_numeric, scale=True):
    """
    Prétraitement :
    1) remplacer les valeurs manquantes par la moyenne ;
    2) standardiser les variables pour éviter qu'une grande échelle domine les distances.

    Exemple :
    si une variable est en milliers et une autre entre 0 et 1,
    la première va dominer le calcul de distance si on ne standardise pas.
    """
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(df_numeric)

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X


def history_to_dataframe(history):
    """
    Convertit l'historique G-Split en DataFrame lisible.
    """
    if not history:
        return pd.DataFrame()

    rows = []

    for d in history:
        rows.append({
            "Tour": d["round"],
            "Cluster-agent": d["cluster"],
            "Taille": d["size"],
            "Décision": "Acceptée" if d["accepted"] else "Refusée",
            "Raison": d["reason"],
            "Inertie avant": d["inertia_before"],
            "Inertie après": d["inertia_after"],
            "Gain inertie": d["inertia_gain"],
            "Silhouette locale": d["silhouette_local"],
            "Signal G-Means": d["gaussian_gap"],
        })

    return pd.DataFrame(rows)


def plot_k_construction(decisions_df):
    """
    Affiche la construction progressive de k.

    k commence à 2.
    Chaque division acceptée ajoute +1 au nombre de clusters.
    """
    if decisions_df.empty:
        return None

    accepted = decisions_df[decisions_df["Décision"] == "Acceptée"]

    k_values = [2]
    steps = [0]

    current_k = 2

    for i in range(len(accepted)):
        current_k += 1
        k_values.append(current_k)
        steps.append(i + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, k_values, marker="o")
    ax.set_title("Construction progressive du nombre de clusters k")
    ax.set_xlabel("Nombre de divisions acceptées")
    ax.set_ylabel("k courant")
    ax.grid(True, alpha=0.3)

    return fig


def plot_split_criteria(decisions_df):
    """
    Affiche les critères des divisions testées :
    - gain d'inertie,
    - silhouette locale,
    - signal inspiré de G-Means.

    Ce graphique aide à justifier pourquoi une division est acceptée ou refusée.
    """
    if decisions_df.empty:
        return None

    df = decisions_df.copy()
    df = df.dropna(subset=["Gain inertie", "Silhouette locale", "Signal G-Means"])

    if df.empty:
        return None

    labels = [f"T{int(r['Tour'])}-C{int(r['Cluster-agent'])}" for _, r in df.iterrows()]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, df["Gain inertie"], marker="o", label="Gain inertie")
    ax.plot(x, df["Silhouette locale"], marker="o", label="Silhouette locale")
    ax.plot(x, df["Signal G-Means"], marker="o", label="Signal G-Means")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Critères utilisés pour accepter/refuser les divisions")
    ax.set_ylabel("Valeur du critère")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def plot_agent_convergence(model):
    """
    Affiche la convergence après le choix de k :
    - inertie globale,
    - transferts entre agents,
    - outliers.
    """
    history = model.history_

    fig, ax = plt.subplots(figsize=(8, 4))

    iterations = np.arange(1, len(history["inertia"]) + 1)
    ax.plot(iterations, history["inertia"], marker="o")

    ax.set_title("Convergence : évolution de l'inertie globale")
    ax.set_xlabel("Itération")
    ax.set_ylabel("Inertie")
    ax.grid(True, alpha=0.3)

    return fig


def plot_transfers(model):
    """
    Affiche le nombre de transferts de points faibles entre agents.
    """
    history = model.history_

    fig, ax = plt.subplots(figsize=(8, 4))

    iterations = np.arange(1, len(history["transfers"]) + 1)
    ax.bar(iterations, history["transfers"])

    ax.set_title("Transferts de points faibles entre agents")
    ax.set_xlabel("Itération")
    ax.set_ylabel("Nombre de transferts")
    ax.grid(True, axis="y", alpha=0.3)

    return fig


def plot_2d_without_pca(X_original_numeric, labels, feature_names):
    """
    Visualisation simple sans ACP.

    Si le dataset contient au moins deux colonnes numériques,
    on affiche les deux premières colonnes numériques.
    Ce n'est pas utilisé pour choisir k.
    C'est uniquement une visualisation optionnelle.
    """
    if X_original_numeric.shape[1] < 2:
        return None

    x_col = feature_names[0]
    y_col = feature_names[1]

    fig, ax = plt.subplots(figsize=(8, 5))

    unique_labels = sorted(np.unique(labels))

    for lab in unique_labels:
        mask = labels == lab

        if lab == -1:
            ax.scatter(
                X_original_numeric.loc[mask, x_col],
                X_original_numeric.loc[mask, y_col],
                marker="x",
                s=70,
                label="Outliers",
            )
        else:
            ax.scatter(
                X_original_numeric.loc[mask, x_col],
                X_original_numeric.loc[mask, y_col],
                s=35,
                alpha=0.75,
                label=f"Cluster {lab}",
            )

    ax.set_title("Visualisation simple avec deux variables originales, sans ACP")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig


def compare_with_classic_kmeans(X, k, random_state):
    """
    Lance un K-means classique avec le même k final pour comparaison.
    """
    km = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )

    labels = km.fit_predict(X)
    return km, labels


# =============================================================================
# BARRE LATERALE
# =============================================================================

with st.sidebar:
    st.header("Paramètres")

    csv_files = sorted(glob.glob("*.csv"))

    if not csv_files:
        st.error("Aucun fichier CSV trouvé dans le dossier courant.")
        st.stop()

    selected_file = st.selectbox("Dataset CSV", csv_files)

    st.divider()

    st.subheader("Prétraitement")
    scale_data = st.checkbox("Standardiser les données", value=True)

    st.divider()

    st.subheader("G-Split Auto-K")

    k_initial = st.number_input(
        "k initial",
        min_value=2,
        max_value=20,
        value=2,
        step=1,
        help="Point de départ. Ce n'est pas le k final."
    )

    k_max = st.number_input(
        "k maximal de sécurité",
        min_value=2,
        max_value=50,
        value=10,
        step=1,
        help="Limite de sécurité pour éviter la sur-division. Ce n'est pas une grid search."
    )

    min_cluster_size = st.number_input(
        "Taille minimale d'un cluster",
        min_value=2,
        max_value=1000,
        value=20,
        step=1,
        help="Un cluster doit être assez grand pour être divisé."
    )

    min_inertia_gain = st.slider(
        "Gain minimal d'inertie",
        min_value=0.01,
        max_value=0.80,
        value=0.20,
        step=0.01,
        help="Exemple : 0.20 signifie réduction minimale de 20% de l'inertie locale."
    )

    min_silhouette_split = st.slider(
        "Silhouette locale minimale",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.01,
        help="La division doit produire deux sous-clusters suffisamment séparés."
    )

    min_gaussian_gap = st.slider(
        "Signal G-Means minimal",
        min_value=0.0,
        max_value=5.0,
        value=1.00,
        step=0.05,
        help="Signal inspiré de G-Means : vérifie si le cluster semble cacher deux groupes."
    )

    st.divider()

    st.subheader("Agents EAC-KMeans")

    weak_points_ratio = st.slider(
        "Ratio de points faibles",
        min_value=0.05,
        max_value=1.00,
        value=0.20,
        step=0.05,
        help="Pourcentage de points éloignés que chaque agent examine."
    )

    max_iter = st.number_input(
        "Nombre maximal d'itérations",
        min_value=5,
        max_value=500,
        value=80,
        step=5,
    )

    random_state = st.number_input(
        "Random state",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
    )

    run = st.button("Lancer l'algorithme", type="primary", use_container_width=True)


# =============================================================================
# CHARGEMENT DATASET
# =============================================================================

df = load_csv(selected_file)
numeric_df = select_numeric_data(df)

st.subheader("Dataset chargé")

c1, c2, c3 = st.columns(3)
c1.metric("Fichier", selected_file)
c2.metric("Lignes", df.shape[0])
c3.metric("Colonnes numériques utilisées", numeric_df.shape[1])

st.dataframe(df.head(10), use_container_width=True)

if numeric_df.shape[1] < 1:
    st.error("Le dataset doit contenir au moins une colonne numérique.")
    st.stop()

feature_names = numeric_df.columns.tolist()


# =============================================================================
# EXECUTION
# =============================================================================

if run:

    X = preprocess_numeric_data(numeric_df, scale=scale_data)

    model = EACKMeans(
        n_clusters="auto_gsplit",
        k_initial=int(k_initial),
        k_max=int(k_max),
        max_iter=int(max_iter),
        weak_points_ratio=float(weak_points_ratio),
        min_cluster_size=int(min_cluster_size),
        random_state=int(random_state),
        min_inertia_gain=float(min_inertia_gain),
        min_silhouette_split=float(min_silhouette_split),
        min_gaussian_gap=float(min_gaussian_gap),
    )

    try:
        model.fit(X)
    except Exception as e:
        st.error(f"Erreur pendant l'exécution : {e}")
        st.stop()

    best_k = model._actual_k_
    gsplit_info = model.gsplit_info_
    decisions_df = history_to_dataframe(gsplit_info.get("history", []))

    st.divider()
    st.subheader("Résultat principal : choix automatique de k")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("k choisi automatiquement", best_k)
    r2.metric("Divisions acceptées", gsplit_info.get("accepted_splits", 0))
    r3.metric("Itérations agents", model.n_iter_)
    r4.metric("Inertie finale", f"{model.inertia_:.4f}")

    st.success(
        f"L'approche G-Split Agent Auto-K a construit automatiquement k = {best_k}. "
        "Ce k n'a pas été choisi par recherche exhaustive, mais par divisions locales justifiées des clusters-agents."
    )

    st.markdown("""
    ### Pourquoi ce k a été choisi ?

    L'algorithme commence avec `k = 2`.  
    Ensuite, chaque cluster-agent teste s'il doit être divisé en deux.

    Une division est acceptée seulement si :

    1. le cluster est assez grand ;
    2. les deux sous-clusters sont assez grands ;
    3. le gain d'inertie locale est suffisant ;
    4. la silhouette locale montre une bonne séparation ;
    5. le signal inspiré de G-Means indique que le cluster n'est pas simplement un seul groupe homogène.

    Donc, chaque nouveau cluster est créé uniquement lorsqu'il est justifié par les données.
    """)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Décisions Auto-K",
        "Graphiques du choix de k",
        "Convergence des agents",
        "Comparaison K-means",
        "Données annotées"
    ])

    # -------------------------------------------------------------------------
    # ONGLET 1 : décisions
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("### Historique des décisions de division")

        if decisions_df.empty:
            st.info("Aucune décision enregistrée.")
        else:
            st.dataframe(decisions_df, use_container_width=True)

            accepted_df = decisions_df[decisions_df["Décision"] == "Acceptée"]
            refused_df = decisions_df[decisions_df["Décision"] == "Refusée"]

            a1, a2 = st.columns(2)
            a1.metric("Divisions acceptées", len(accepted_df))
            a2.metric("Divisions refusées", len(refused_df))

        st.markdown("""
        **Interprétation pour le professeur :**

        Si une division est refusée, cela signifie que créer un nouveau cluster n'était pas suffisamment justifié.
        Si une division est acceptée, alors `k` augmente de 1.
        """)

    # -------------------------------------------------------------------------
    # ONGLET 2 : graphiques du choix de k
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### Construction progressive de k")

        fig_k = plot_k_construction(decisions_df)
        if fig_k is not None:
            st.pyplot(fig_k)
        else:
            st.info("Pas assez d'informations pour tracer la construction de k.")

        st.markdown("### Critères utilisés pour accepter/refuser les divisions")

        fig_criteria = plot_split_criteria(decisions_df)
        if fig_criteria is not None:
            st.pyplot(fig_criteria)
        else:
            st.info("Aucune division testée avec critères numériques affichables.")

    # -------------------------------------------------------------------------
    # ONGLET 3 : convergence agents
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("### Après le choix de k : comportement des agents")

        st.pyplot(plot_agent_convergence(model))
        st.pyplot(plot_transfers(model))

        st.markdown("""
        Les transferts montrent que les clusters ne sont pas figés.
        Après le choix automatique de `k`, les agents peuvent encore échanger des points faibles
        si cela améliore l'inertie globale.
        """)

    # -------------------------------------------------------------------------
    # ONGLET 4 : comparaison
    # -------------------------------------------------------------------------
    with tab4:
        st.markdown("### Comparaison avec K-means classique au même k")

        classic_model, classic_labels = compare_with_classic_kmeans(
            X,
            k=best_k,
            random_state=int(random_state),
        )

        comparison = pd.DataFrame({
            "Méthode": ["K-means classique", "EAC-KMeans G-Split"],
            "k utilisé": [best_k, best_k],
            "Inertie": [classic_model.inertia_, model.inertia_],
            "Itérations": [classic_model.n_iter_, model.n_iter_],
            "Outliers": [0, int(np.sum(model.outlier_mask_))],
        })

        st.dataframe(comparison, use_container_width=True)

        gain = classic_model.inertia_ - model.inertia_

        if gain > 0:
            st.success(
                f"EAC-KMeans G-Split réduit l'inertie de {100 * gain / classic_model.inertia_:.2f}% "
                "par rapport au K-means classique avec le même k."
            )
        else:
            st.info(
                "Sur ce dataset, l'inertie est proche ou supérieure à celle du K-means classique. "
                "L'intérêt principal reste ici le choix automatique et justifié de k."
            )

        st.markdown("### Visualisation simple sans ACP")

        fig_2d = plot_2d_without_pca(
            numeric_df.reset_index(drop=True),
            model.final_labels_,
            feature_names,
        )

        if fig_2d is not None:
            st.pyplot(fig_2d)
        else:
            st.info("Il faut au moins deux colonnes numériques pour afficher un nuage de points simple.")

    # -------------------------------------------------------------------------
    # ONGLET 5 : export
    # -------------------------------------------------------------------------
    with tab5:
        st.markdown("### Dataset annoté")

        result_df = df.copy()
        result_df["cluster_gsplit_eac"] = model.final_labels_
        result_df["is_outlier"] = model.outlier_mask_.astype(int)

        st.dataframe(result_df, use_container_width=True)

        csv_data = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Télécharger le dataset annoté",
            data=csv_data,
            file_name="resultats_gsplit_eac.csv",
            mime="text/csv",
        )

else:
    st.info("Choisis les paramètres dans la barre latérale puis clique sur « Lancer l'algorithme ».")
