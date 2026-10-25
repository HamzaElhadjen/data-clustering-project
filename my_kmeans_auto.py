
"""
my_kmeans.py

EAC-KMeans avec sélection automatique de k par G-Split Agent Auto-K


Objectif du fichier

Ce fichier contient l'algorithme principal du projet.

L'idée importante :
    - On conserve l'approche multi-agents.
    - Chaque cluster est considéré comme un agent autonome.
    - Le nombre de clusters k n'est pas donné par l'utilisateur.
    - Le nombre k est construit automatiquement par une stratégie G-Split.

Différence avec une grid search

Une grid search teste plusieurs valeurs de k :
    k = 2, k = 3, k = 4, ..., k_max

Notre approche ne fonctionne pas comme ça.

Notre approche :
    1. Commence avec un petit k initial.
    2. Analyse chaque cluster-agent séparément.
    3. Décide si ce cluster-agent mérite d'être divisé.
    4. Accepte une division seulement si elle est justifiée.
    5. Répète jusqu'à stabilisation.

Donc, k émerge progressivement à partir de la structure des données.

Pourquoi G-Split ?

Le nom G-Split vient de :
    - G : inspiration de G-Means.
    - Split : division automatique des clusters.
    - Agent : chaque cluster prend localement une décision.

Un cluster est divisé seulement s'il semble contenir plusieurs sous-groupes.
"""

import time
import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_array, check_is_fitted

# mesure  de  securite  du  choix  de  k
AUTO_K_INITIAL = 2
AUTO_K_MAX = 100

AUTO_MIN_CLUSTER_SIZE = 20

# Une division doit réduire au moins 20% de l'inertie locale.
AUTO_MIN_INERTIA_GAIN = 0.20

# Une division doit produire deux sous-clusters suffisamment séparés.
AUTO_MIN_SILHOUETTE_SPLIT = 0.35

# Signal inspiré de G-Means.
# Plus la valeur est grande, plus on exige une séparation claire.
AUTO_MIN_GAUSSIAN_GAP = 1.00

# Paramètres des agents
AUTO_MAX_ITER = 80
AUTO_TOL = 1e-4
AUTO_WEAK_POINTS_RATIO = 0.20
AUTO_EXCHANGE_TOLERANCE = 1e-8
AUTO_OUTLIER_FACTOR = 2.5
AUTO_REASSIGN_OUTLIERS = True
AUTO_RANDOM_STATE = 42

# OUTILS MATHEMATIQUES

def _sse(X):
    """
    Calcule l'inertie locale d'un ensemble de points.

    Inertie = somme des distances au carré entre les points et leur centroïde.

    Rôle dans le projet :
        - mesurer la compacité d'un cluster ;
        - vérifier si une division rend les groupes plus compacts ;
        - calculer le gain d'inertie locale.

    Exemple :
        Si l'inertie baisse fortement après une division, cela veut dire que les
        deux sous-clusters sont plus compacts que le cluster initial.
    """
    if len(X) == 0:
        return 0.0

    center = X.mean(axis=0)
    return float(np.sum((X - center) ** 2))


def _safe_silhouette(X, labels):
    """
    Calcule la silhouette locale d'une division.

    Silhouette :
        proche de 1  -> bonne séparation
        proche de 0  -> séparation faible
        négative     -> mauvaise séparation

    Pourquoi on l'utilise ?
        L'inertie diminue presque toujours quand on augmente k.
        Donc l'inertie seule peut encourager trop de divisions.
        La silhouette ajoute une condition de séparation entre les groupes.
    """
    unique = np.unique(labels)

    if len(unique) < 2:
        return -1.0

    if len(X) <= len(unique):
        return -1.0

    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return -1.0


def _gaussian_gap_score(projected_values):
    """
    Calcule un signal simple inspiré de G-Means.

    Idée de G-Means :
        Un cluster valide ressemble souvent à une seule masse gaussienne.
        Si un cluster cache deux groupes, sa projection sur l'axe de séparation
        montre souvent un grand espace vide entre deux zones denses.

    Méthode simplifiée :
        1. Trier les valeurs projetées.
        2. Chercher le plus grand espace entre deux valeurs consécutives.
        3. Diviser cet espace par l'écart-type global.

    Interprétation :
        score faible -> le cluster semble homogène ;
        score élevé  -> le cluster semble contenir deux groupes.
    """
    values = np.asarray(projected_values, dtype=float)

    if len(values) < 3:
        return 0.0

    std = np.std(values)

    if std < 1e-12:
        return 0.0

    sorted_values = np.sort(values)
    gaps = np.diff(sorted_values)

    if len(gaps) == 0:
        return 0.0

    return float(np.max(gaps) / (std + 1e-12))


def _gmeans_signal(X_cluster, sub_centers):
    """
    Calcule le signal G-Means d'un cluster candidat à la division.

    Étapes :
        1. On prend les deux sous-centroïdes obtenus par KMeans local avec k=2.
        2. On construit l'axe allant du premier sous-centroïde au deuxième.
        3. On projette tous les points du cluster sur cet axe.
        4. On cherche si cette projection montre une séparation claire.

    Ce signal ne choisit pas k directement.
    Il décide seulement si un cluster-agent peut se diviser.
    """
    c1 = sub_centers[0]
    c2 = sub_centers[1]

    direction = c2 - c1
    norm = np.linalg.norm(direction)

    if norm < 1e-12:
        return 0.0

    direction = direction / norm
    projected = X_cluster @ direction

    return _gaussian_gap_score(projected)

# G-SPLIT AGENT AUTO-K


def select_k_by_gsplit_auto(X):
    """
    Sélectionne automatiquement k avec l'approche G-Split Agent Auto-K.

    Cette fonction est le cœur de l'amélioration du projet.

    Principe :
        - commencer avec k = 2 ;
        - chaque cluster est un agent ;
        - chaque agent teste une division locale en deux ;
        - la division est acceptée seulement si plusieurs critères sont validés ;
        - chaque division acceptée augmente k de 1 ;
        - l'algorithme s'arrête quand aucun agent ne peut se diviser.

    Critères d'acceptation d'une division :
        1. Le cluster initial est assez grand.
        2. Les deux sous-clusters sont assez grands.
        3. Le gain d'inertie locale est suffisant.
        4. La silhouette locale est suffisante.
        5. Le signal G-Means indique que le cluster n'est pas homogène.

    Retour :
        best_k : le nombre de clusters trouvé automatiquement.
        info   : dictionnaire contenant l'historique des décisions.
    """
    X = check_array(X, dtype=float)

    n_samples = X.shape[0]

    if n_samples < 2:
        raise ValueError("Il faut au moins deux points pour faire du clustering.")

    k_current = min(AUTO_K_INITIAL, n_samples)
    k_max = min(AUTO_K_MAX, n_samples)

    start_time = time.time()

    # Première partition avec un petit k.
    base = KMeans(
        n_clusters=k_current,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=AUTO_RANDOM_STATE,
    )

    labels = base.fit_predict(X)

    history = []
    accepted_splits = 0

    # On limite le nombre de tours pour éviter toute boucle excessive.
    max_rounds = AUTO_K_MAX

    for round_id in range(1, max_rounds + 1):

        splits_this_round = 0

        # Copie modifiable des labels.
        new_labels = labels.copy()

        # Le prochain identifiant de cluster disponible.
        next_label = int(new_labels.max()) + 1

        # Liste des clusters actuels.
        current_clusters = sorted(np.unique(labels))

        for cluster_id in current_clusters:

            # k_max est une sécurité, pas une grid search.
            if len(np.unique(new_labels)) >= k_max:
                break

            indices = np.where(labels == cluster_id)[0]
            Xc = X[indices]
            n_c = len(Xc)

            decision = {
                "round": int(round_id),
                "cluster": int(cluster_id),
                "size": int(n_c),
                "accepted": False,
                "reason": "",
                "inertia_before": np.nan,
                "inertia_after": np.nan,
                "inertia_gain": np.nan,
                "silhouette_local": np.nan,
                "gaussian_signal": np.nan,
                "subcluster_1_size": np.nan,
                "subcluster_2_size": np.nan,
            }

            # Critère 1 : taille minimale du cluster à diviser.
            if n_c < 2 * AUTO_MIN_CLUSTER_SIZE:
                decision["reason"] = "Refus : cluster trop petit"
                history.append(decision)
                continue

            inertia_before = _sse(Xc)
            decision["inertia_before"] = inertia_before

            if inertia_before < 1e-12:
                decision["reason"] = "Refus : cluster déjà très compact"
                history.append(decision)
                continue

            # Test local : l'agent essaie de se diviser en deux.
            local = KMeans(
                n_clusters=2,
                init="k-means++",
                n_init=10,
                max_iter=200,
                random_state=AUTO_RANDOM_STATE + round_id + int(cluster_id),
            )

            local_labels = local.fit_predict(Xc)
            counts = np.bincount(local_labels)

            # Critère 2 : les deux sous-clusters doivent être assez grands.
            if len(counts) < 2 or np.min(counts) < AUTO_MIN_CLUSTER_SIZE:
                decision["reason"] = "Refus : sous-cluster trop petit"
                if len(counts) >= 2:
                    decision["subcluster_1_size"] = int(counts[0])
                    decision["subcluster_2_size"] = int(counts[1])
                history.append(decision)
                continue

            decision["subcluster_1_size"] = int(counts[0])
            decision["subcluster_2_size"] = int(counts[1])

            X1 = Xc[local_labels == 0]
            X2 = Xc[local_labels == 1]

            inertia_after = _sse(X1) + _sse(X2)
            inertia_gain = (inertia_before - inertia_after) / (inertia_before + 1e-12)

            silhouette_local = _safe_silhouette(Xc, local_labels)
            gaussian_signal = _gmeans_signal(Xc, local.cluster_centers_)

            decision["inertia_after"] = float(inertia_after)
            decision["inertia_gain"] = float(inertia_gain)
            decision["silhouette_local"] = float(silhouette_local)
            decision["gaussian_signal"] = float(gaussian_signal)

            # Critère 3 : gain d'inertie.
            if inertia_gain < AUTO_MIN_INERTIA_GAIN:
                decision["reason"] = "Refus : gain d'inertie insuffisant"
                history.append(decision)
                continue

            # Critère 4 : séparation locale.
            if silhouette_local < AUTO_MIN_SILHOUETTE_SPLIT:
                decision["reason"] = "Refus : silhouette locale insuffisante"
                history.append(decision)
                continue

            # Critère 5 : inspiration G-Means.
            if gaussian_signal < AUTO_MIN_GAUSSIAN_GAP:
                decision["reason"] = "Refus : signal G-Means insuffisant"
                history.append(decision)
                continue

            # Tous les critères sont satisfaits : division acceptée.
            global_sub_0 = indices[local_labels == 0]
            global_sub_1 = indices[local_labels == 1]

            new_labels[global_sub_0] = cluster_id
            new_labels[global_sub_1] = next_label
            next_label += 1

            decision["accepted"] = True
            decision["reason"] = "Acceptée : division justifiée"

            history.append(decision)

            accepted_splits += 1
            splits_this_round += 1

        labels = new_labels.copy()

        # Arrêt naturel : aucun agent n'a réussi à justifier une division.
        if splits_this_round == 0:
            break

    # Renumérotation propre : 0, 1, 2, ...
    unique_labels = sorted(np.unique(labels))
    mapping = {old: new for new, old in enumerate(unique_labels)}
    final_labels = np.array([mapping[v] for v in labels], dtype=int)

    best_k = len(np.unique(final_labels))

    elapsed_time = time.time() - start_time

    info = {
        "method": "G-Split Agent Auto-K",
        "best_k": int(best_k),
        "labels": final_labels,
        "history": history,
        "accepted_splits": int(accepted_splits),
        "selection_time": float(elapsed_time),
        "parameters": {
            "k_initial": AUTO_K_INITIAL,
            "k_max_security": AUTO_K_MAX,
            "min_cluster_size": AUTO_MIN_CLUSTER_SIZE,
            "min_inertia_gain": AUTO_MIN_INERTIA_GAIN,
            "min_silhouette_split": AUTO_MIN_SILHOUETTE_SPLIT,
            "min_gaussian_gap": AUTO_MIN_GAUSSIAN_GAP,
        },
    }

    return best_k, info

# EAC-KMEANS : AGENTS + ECHANGES

class EACKMeans(BaseEstimator, ClusterMixin):
    """
    EAC-KMeans final.

    Ce modèle fait deux choses :
        1. Choisir k automatiquement avec G-Split Agent Auto-K.
        2. Optimiser les clusters avec une logique multi-agents.

    Logique multi-agents :
        - Chaque cluster est un agent.
        - Chaque agent détecte ses points faibles.
        - Un point faible peut être transféré à un autre agent.
        - Le transfert est accepté uniquement s'il réduit l'inertie globale.
        - Les points très éloignés peuvent être marqués comme outliers.

    L'utilisateur ne règle rien :
        tous les paramètres sont automatiques.
    """

    def __init__(self):
        # Aucun paramètre exposé à l'utilisateur.
        # Cela force l'automatisation complète de l'algorithme.
        self.n_clusters = "auto_gsplit"

    def _distances_squared(self, X, centers):
        """
        Calcule la matrice des distances au carré.
        Chaque ligne correspond à un point.
        Chaque colonne correspond à un centroïde.
        """
        return np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    def _compute_centers(self, X, labels, outlier_mask):
        """
        Recalcule les centroïdes des agents.
        Les outliers sont ignorés pour ne pas déplacer artificiellement les centres.
        """
        centers = []

        rng = np.random.default_rng(AUTO_RANDOM_STATE)

        for c in range(self._actual_k_):
            mask = (labels == c) & (~outlier_mask)

            if np.sum(mask) == 0:
                centers.append(X[rng.integers(0, len(X))])
            else:
                centers.append(X[mask].mean(axis=0))

        return np.array(centers)

    def _compute_inertia(self, X, labels, centers, outlier_mask):
        """
        Calcule l'inertie globale du modèle.
        Plus elle est faible, plus les clusters sont compacts.
        """
        total = 0.0

        for c in range(self._actual_k_):
            mask = (labels == c) & (~outlier_mask)
            if np.any(mask):
                total += np.sum((X[mask] - centers[c]) ** 2)

        return float(total)

    def _find_weak_points(self, X, labels, centers, outlier_mask):
        """
        Chaque agent cherche ses points faibles.

        Définition simple :
            un point faible est un point parmi les plus éloignés du centroïde
            de son propre cluster.

        Ces points sont les meilleurs candidats pour un transfert.
        """
        weak_indices = []

        for c in range(self._actual_k_):
            indices = np.where((labels == c) & (~outlier_mask))[0]

            if len(indices) <= AUTO_MIN_CLUSTER_SIZE:
                continue

            Xc = X[indices]
            distances = np.sum((Xc - centers[c]) ** 2, axis=1)

            n_weak = max(1, int(np.ceil(AUTO_WEAK_POINTS_RATIO * len(indices))))
            weakest_local = np.argsort(distances)[-n_weak:]

            weak_indices.extend(indices[weakest_local])

        return weak_indices

    def _try_transfer_point(self, X, labels, centers, outlier_mask, index):
        """
        Essaie de transférer un point faible vers un autre agent.

        Règle :
            si le transfert réduit l'inertie globale, on accepte ;
            sinon, on refuse.

        Cette règle évite les transferts arbitraires.
        """
        current_cluster = labels[index]

        if outlier_mask[index]:
            return False

        if np.sum((labels == current_cluster) & (~outlier_mask)) <= AUTO_MIN_CLUSTER_SIZE:
            return False

        old_inertia = self._compute_inertia(X, labels, centers, outlier_mask)

        x = X[index]
        distances_to_centers = np.sum((centers - x) ** 2, axis=1)
        candidate_clusters = np.argsort(distances_to_centers)

        for target_cluster in candidate_clusters:

            if target_cluster == current_cluster:
                continue

            new_labels = labels.copy()
            new_labels[index] = target_cluster

            new_centers = self._compute_centers(X, new_labels, outlier_mask)
            new_inertia = self._compute_inertia(X, new_labels, new_centers, outlier_mask)

            if new_inertia < old_inertia - AUTO_EXCHANGE_TOLERANCE:
                labels[index] = target_cluster
                return True

        return False

    def _detect_outliers(self, X, labels, centers, outlier_mask):
        """
        Détecte les points très éloignés dans chaque cluster.

        Un point est considéré comme outlier si sa distance au centroïde dépasse :
            moyenne_distance + facteur * écart_type_distance

        Cette étape rend l'algorithme plus robuste aux points isolés.
        """
        detected = 0

        for c in range(self._actual_k_):
            indices = np.where((labels == c) & (~outlier_mask))[0]

            if len(indices) <= AUTO_MIN_CLUSTER_SIZE:
                continue

            Xc = X[indices]
            distances = np.sqrt(np.sum((Xc - centers[c]) ** 2, axis=1))

            threshold = np.mean(distances) + AUTO_OUTLIER_FACTOR * np.std(distances)

            candidate_outliers = indices[distances > threshold]

            for idx in candidate_outliers:
                if np.sum((labels == c) & (~outlier_mask)) > AUTO_MIN_CLUSTER_SIZE:
                    outlier_mask[idx] = True
                    detected += 1

        return detected

    def fit(self, X, y=None):
        """
        Entraîne l'algorithme complet.

        Étape 1 :
            choix automatique de k avec G-Split.

        Étape 2 :
            initialisation KMeans avec le k obtenu.

        Étape 3 :
            optimisation multi-agents :
                - assignation,
                - recherche des points faibles,
                - transferts bénéfiques,
                - détection d'outliers,
                - arrêt automatique.
        """
        X = check_array(X, dtype=float)

        fit_start_time = time.time()

        # ---------------------------------------------------------------------
        # 1. Sélection automatique de k
        # ---------------------------------------------------------------------
        best_k, gsplit_info = select_k_by_gsplit_auto(X)

        self._actual_k_ = int(best_k)
        self.gsplit_info_ = gsplit_info

        # 2. KMeans initial avec le k trouvé automatiquement
        baseline_same_k = KMeans(
            n_clusters=self._actual_k_,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=AUTO_RANDOM_STATE,
        )

        labels = baseline_same_k.fit_predict(X)
        centers = baseline_same_k.cluster_centers_.copy()
        baseline_inertia_same_k = float(baseline_same_k.inertia_)

        outlier_mask = np.zeros(X.shape[0], dtype=bool)

        self.history_ = {
            "inertia": [],
            "transfers": [],
            "outliers": [],
        }

        previous_inertia = np.inf

        # 3. Optimisation par agents
        for iteration in range(1, AUTO_MAX_ITER + 1):

            centers = self._compute_centers(X, labels, outlier_mask)

            # Assignation standard.
            non_outliers = np.where(~outlier_mask)[0]

            if len(non_outliers) > 0:
                distances = self._distances_squared(X[non_outliers], centers)
                labels[non_outliers] = np.argmin(distances, axis=1)

            centers = self._compute_centers(X, labels, outlier_mask)

            # Recherche des points faibles.
            weak_points = self._find_weak_points(X, labels, centers, outlier_mask)

            transfers = 0

            for idx in weak_points:
                moved = self._try_transfer_point(X, labels, centers, outlier_mask, idx)

                if moved:
                    transfers += 1
                    centers = self._compute_centers(X, labels, outlier_mask)

            detected_outliers = self._detect_outliers(X, labels, centers, outlier_mask)
            centers = self._compute_centers(X, labels, outlier_mask)

            inertia = self._compute_inertia(X, labels, centers, outlier_mask)

            self.history_["inertia"].append(float(inertia))
            self.history_["transfers"].append(int(transfers))
            self.history_["outliers"].append(int(np.sum(outlier_mask)))

            improvement = previous_inertia - inertia

            if improvement >= 0 and improvement < AUTO_TOL and transfers == 0 and detected_outliers == 0:
                break

            previous_inertia = inertia

        total_time = time.time() - fit_start_time

        self.cluster_centers_ = centers
        self.labels_ = labels.copy()
        self.outlier_mask_ = outlier_mask.copy()

        self.final_labels_ = labels.copy()
        self.final_labels_[outlier_mask] = -1

        self.inertia_ = float(self.history_["inertia"][-1])
        self.n_iter_ = len(self.history_["inertia"])

        # Métriques de gain et d'efficacité.
        inertia_gain = baseline_inertia_same_k - self.inertia_
        if baseline_inertia_same_k > 1e-12:
            inertia_gain_percent = 100.0 * inertia_gain / baseline_inertia_same_k
        else:
            inertia_gain_percent = 0.0

        self.optimization_report_ = {
            "baseline_inertia_same_k": float(baseline_inertia_same_k),
            "final_inertia": float(self.inertia_),
            "inertia_gain": float(inertia_gain),
            "inertia_gain_percent": float(inertia_gain_percent),
            "total_transfers": int(np.sum(self.history_["transfers"])),
            "outliers": int(np.sum(outlier_mask)),
            "iterations": int(self.n_iter_),
            "selection_time": float(gsplit_info["selection_time"]),
            "total_time": float(total_time),
            "accepted_splits": int(gsplit_info["accepted_splits"]),
        }

        return self

    def predict(self, X):
        """
        Prédit le cluster de nouveaux points.
        """
        check_is_fitted(self, ["cluster_centers_"])
        X = check_array(X, dtype=float)

        distances = self._distances_squared(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):
        """
        Entraîne le modèle et retourne les labels.
        -1 signifie outlier.
        """
        self.fit(X, y)
        return self.final_labels_
