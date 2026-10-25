
"""
my_kmeans.py
Projet : EAC-KMeans avec choix automatique de k par G-Split Agent Auto-K.

Idée principale :
- Chaque cluster est considéré comme un agent.
- Au lieu de tester k=2,3,4,... comme une grid search, on commence avec k=2.
- Chaque agent-cluster analyse sa structure interne.
- Si un cluster semble contenir deux sous-groupes réels, il peut se diviser.
- Une division est acceptée uniquement si elle respecte plusieurs critères :
    1) taille minimale,
    2) gain d'inertie locale,
    3) silhouette locale,
    4) test d'homogénéité inspiré de G-Means.
- Ensuite, l'algorithme EAC-KMeans applique son idée d'agents :
    les points faibles peuvent être transférés entre clusters si cela réduit l'inertie.

Cette version est volontairement claire et pédagogique pour être expliquée au professeur.
"""

import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils.validation import check_array, check_is_fitted


# =============================================================================
# OUTILS MATHEMATIQUES SIMPLES
# =============================================================================

def _sse(X):
    """
    Calcule l'inertie locale d'un ensemble de points.

    Inertie locale = somme des distances au carré entre les points et leur centre.

    Si l'inertie est faible :
        les points sont proches du centre, donc le cluster est compact.

    Si l'inertie est forte :
        les points sont dispersés, donc le cluster peut cacher plusieurs groupes.
    """
    if len(X) == 0:
        return 0.0
    center = X.mean(axis=0)
    return float(np.sum((X - center) ** 2))


def _safe_silhouette(X, labels):
    """
    Calcule la silhouette locale d'une division en deux sous-clusters.

    La silhouette mesure si les deux sous-groupes sont bien séparés :
        proche de 1  : très bonne séparation
        proche de 0  : séparation faible
        négative     : mauvais découpage

    On utilise une fonction sécurisée pour éviter les erreurs quand les données
    sont trop petites ou quand un seul label existe.
    """
    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        return -1.0

    if len(X) <= len(unique_labels):
        return -1.0

    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return -1.0


def _anderson_like_gaussian_score(values):
    """
    Test simplifié inspiré de G-Means.

    G-Means vérifie si les points d'un cluster ressemblent à une seule distribution
    gaussienne. Ici, pour rester simple et facile à défendre, on utilise une idée
    proche :

    1) On projette les points du cluster sur l'axe qui sépare les deux sous-centres.
    2) Si le cluster est vraiment un seul groupe, la projection doit ressembler
       à une seule masse compacte.
    3) Si la projection montre deux zones séparées, le cluster est probablement
       composé de deux groupes.

    Cette fonction renvoie un score de séparation normalisé :
        score élevé  -> séparation claire -> division possible
        score faible -> un seul groupe     -> division refusée
    """
    values = np.asarray(values, dtype=float)

    if len(values) < 3:
        return 0.0

    std = np.std(values)

    if std < 1e-12:
        return 0.0

    # On trie la projection et on cherche le plus grand "trou" entre deux points.
    sorted_values = np.sort(values)
    gaps = np.diff(sorted_values)

    if len(gaps) == 0:
        return 0.0

    max_gap = np.max(gaps)

    # Score sans unité : taille du plus grand trou / écart-type global.
    return float(max_gap / (std + 1e-12))


def _gmeans_split_signal(X_cluster, sub_centers):
    """
    Calcule le signal inspiré de G-Means.

    On prend les deux sous-centroïdes produits par KMeans(k=2).
    L'axe entre ces deux centres est la direction la plus naturelle pour vérifier
    si le cluster cache deux groupes.

    Ensuite, on projette tous les points sur cet axe.
    Si la projection montre une séparation importante, on considère que le cluster
    n'est pas homogène et qu'une division peut être justifiée.
    """
    c1, c2 = sub_centers[0], sub_centers[1]
    direction = c2 - c1
    norm = np.linalg.norm(direction)

    if norm < 1e-12:
        return 0.0

    direction = direction / norm
    projections = X_cluster @ direction

    return _anderson_like_gaussian_score(projections)


# =============================================================================
# G-SPLIT AUTO-K
# =============================================================================

def select_k_by_gsplit(
    X,
    k_initial=2,
    k_max=10,
    min_cluster_size=20,
    min_inertia_gain=0.20,
    min_silhouette_split=0.35,
    min_gaussian_gap=1.00,
    random_state=42,
    max_rounds=10,
):
    """
    Choisit automatiquement k par l'approche G-Split Agent Auto-K.

    Différence avec une grid search :
    --------------------------------
    On ne teste pas toutes les valeurs de k pour choisir la meilleure.
    On commence avec k_initial, puis les clusters se divisent progressivement
    seulement si leur structure interne le justifie.

    Paramètres importants :
    -----------------------
    k_initial :
        Nombre initial de clusters. Généralement 2.

    k_max :
        Limite de sécurité. Ce n'est pas une recherche exhaustive.
        Cela évite seulement de créer trop de clusters.

    min_cluster_size :
        Taille minimale pour autoriser une division.
        Un cluster trop petit ne donne pas une décision fiable.

    min_inertia_gain :
        Gain minimal d'inertie locale.
        Exemple : 0.20 signifie que la division doit réduire au moins 20%
        de l'inertie du cluster.

    min_silhouette_split :
        Qualité minimale de séparation entre les deux sous-clusters.

    min_gaussian_gap :
        Signal minimal inspiré de G-Means.
        Plus il est élevé, plus on demande une séparation claire dans le cluster.

    Retour :
    --------
    best_k : int
        Nombre de clusters construit automatiquement.

    info : dict
        Historique détaillé des décisions pour l'interface et le rapport.
    """
    X = check_array(X, dtype=float)

    n_samples = X.shape[0]

    if n_samples < 2:
        raise ValueError("Il faut au moins 2 points pour faire du clustering.")

    k_current = max(2, int(k_initial))
    k_max = min(int(k_max), n_samples)

    # Première partition simple.
    km = KMeans(
        n_clusters=k_current,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=random_state,
    )
    labels = km.fit_predict(X)

    history = []
    accepted_total = 0

    for round_id in range(1, max_rounds + 1):

        accepted_this_round = 0
        new_labels = labels.copy()
        next_label = int(new_labels.max()) + 1

        # On parcourt les clusters existants.
        current_clusters = sorted(np.unique(labels))

        for cluster_id in current_clusters:

            # Stop si on a atteint la limite de sécurité.
            if len(np.unique(new_labels)) >= k_max:
                break

            indices = np.where(labels == cluster_id)[0]
            Xc = X[indices]
            n_c = len(Xc)

            decision = {
                "round": round_id,
                "cluster": int(cluster_id),
                "size": int(n_c),
                "accepted": False,
                "reason": "",
                "inertia_before": None,
                "inertia_after": None,
                "inertia_gain": None,
                "silhouette_local": None,
                "gaussian_gap": None,
            }

            # Critère 1 : taille minimale.
            if n_c < 2 * min_cluster_size:
                decision["reason"] = "refus : cluster trop petit pour une division fiable"
                history.append(decision)
                continue

            inertia_before = _sse(Xc)
            decision["inertia_before"] = inertia_before

            if inertia_before < 1e-12:
                decision["reason"] = "refus : cluster déjà très compact"
                history.append(decision)
                continue

            # Test local : on essaie de diviser ce cluster en deux.
            local_km = KMeans(
                n_clusters=2,
                init="k-means++",
                n_init=10,
                max_iter=200,
                random_state=random_state + round_id + int(cluster_id),
            )
            local_labels = local_km.fit_predict(Xc)

            # Critère 2 : les deux sous-clusters doivent être assez grands.
            counts = np.bincount(local_labels)
            if len(counts) < 2 or np.min(counts) < min_cluster_size:
                decision["reason"] = "refus : un sous-cluster est trop petit"
                history.append(decision)
                continue

            Xc1 = Xc[local_labels == 0]
            Xc2 = Xc[local_labels == 1]
            inertia_after = _sse(Xc1) + _sse(Xc2)

            inertia_gain = (inertia_before - inertia_after) / (inertia_before + 1e-12)
            silhouette_local = _safe_silhouette(Xc, local_labels)
            gaussian_gap = _gmeans_split_signal(Xc, local_km.cluster_centers_)

            decision["inertia_after"] = inertia_after
            decision["inertia_gain"] = float(inertia_gain)
            decision["silhouette_local"] = float(silhouette_local)
            decision["gaussian_gap"] = float(gaussian_gap)

            # Critère 3 : gain d'inertie.
            if inertia_gain < min_inertia_gain:
                decision["reason"] = "refus : gain d'inertie insuffisant"
                history.append(decision)
                continue

            # Critère 4 : silhouette locale.
            if silhouette_local < min_silhouette_split:
                decision["reason"] = "refus : séparation locale insuffisante"
                history.append(decision)
                continue

            # Critère 5 : signal inspiré de G-Means.
            if gaussian_gap < min_gaussian_gap:
                decision["reason"] = "refus : le cluster semble encore homogène"
                history.append(decision)
                continue

            # Si tous les critères sont validés, on accepte la division.
            global_sub0 = indices[local_labels == 0]
            global_sub1 = indices[local_labels == 1]

            # On garde l'ancien identifiant pour le premier sous-cluster
            # et on crée un nouveau label pour le deuxième.
            new_labels[global_sub0] = cluster_id
            new_labels[global_sub1] = next_label
            next_label += 1

            decision["accepted"] = True
            decision["reason"] = "acceptée : division justifiée par les critères"
            history.append(decision)

            accepted_this_round += 1
            accepted_total += 1

        labels = new_labels.copy()

        # Si aucun agent-cluster ne se divise, k est stabilisé.
        if accepted_this_round == 0:
            break

    # Renumérotation propre des labels : 0,1,2,...
    unique = sorted(np.unique(labels))
    mapping = {old: new for new, old in enumerate(unique)}
    final_labels = np.array([mapping[v] for v in labels], dtype=int)
    best_k = len(np.unique(final_labels))

    info = {
        "method": "G-Split Agent Auto-K",
        "best_k": int(best_k),
        "labels": final_labels,
        "history": history,
        "accepted_splits": int(accepted_total),
        "parameters": {
            "k_initial": k_initial,
            "k_max": k_max,
            "min_cluster_size": min_cluster_size,
            "min_inertia_gain": min_inertia_gain,
            "min_silhouette_split": min_silhouette_split,
            "min_gaussian_gap": min_gaussian_gap,
        },
    }

    return best_k, info


# =============================================================================
# EAC-KMEANS : CLUSTERS COMME AGENTS + TRANSFERT DE POINTS FAIBLES
# =============================================================================

class EACKMeans(BaseEstimator, ClusterMixin):
    """
    EAC-KMeans : Exchange-Aware Agent Clustering K-Means.

    L'idée des agents est conservée :
    ---------------------------------
    - Chaque cluster est un agent.
    - Après une étape K-means classique, chaque agent détecte ses points faibles.
    - Un point faible est un point éloigné du centre de son cluster.
    - Un agent peut transférer un point faible vers un autre agent si cela réduit
      l'inertie globale.
    - Les points très éloignés peuvent être marqués comme outliers.

    Choix automatique de k :
    ------------------------
    n_clusters="auto_gsplit"
        utilise la nouvelle approche G-Split Agent Auto-K.

    n_clusters=int
        utilise directement le k donné.
    """

    def __init__(
        self,
        n_clusters="auto_gsplit",
        k_initial=2,
        k_max=10,
        max_iter=100,
        tol=1e-4,
        weak_points_ratio=0.20,
        min_cluster_size=20,
        exchange_tolerance=1e-8,
        outlier_factor=2.5,
        reassign_outliers=True,
        random_state=42,
        # paramètres G-Split
        min_inertia_gain=0.20,
        min_silhouette_split=0.35,
        min_gaussian_gap=1.00,
    ):
        self.n_clusters = n_clusters
        self.k_initial = k_initial
        self.k_max = k_max
        self.max_iter = max_iter
        self.tol = tol
        self.weak_points_ratio = weak_points_ratio
        self.min_cluster_size = min_cluster_size
        self.exchange_tolerance = exchange_tolerance
        self.outlier_factor = outlier_factor
        self.reassign_outliers = reassign_outliers
        self.random_state = random_state
        self.min_inertia_gain = min_inertia_gain
        self.min_silhouette_split = min_silhouette_split
        self.min_gaussian_gap = min_gaussian_gap

    def _distances_squared(self, X, centers):
        """
        Calcule les distances euclidiennes au carré entre chaque point et chaque centre.
        Résultat : matrice de taille (n_points, n_clusters).
        """
        return np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    def _compute_centers(self, X, labels, outlier_mask):
        """
        Recalcule le centre de chaque cluster avec les points non-outliers.
        """
        centers = []

        for c in range(self._actual_k_):
            mask = (labels == c) & (~outlier_mask)

            if np.sum(mask) == 0:
                # Si un cluster devient vide, on choisit un point aléatoire.
                rng = np.random.default_rng(self.random_state)
                centers.append(X[rng.integers(0, len(X))])
            else:
                centers.append(X[mask].mean(axis=0))

        return np.array(centers)

    def _compute_inertia(self, X, labels, centers, outlier_mask):
        """
        Calcule l'inertie globale en ignorant les outliers.
        """
        total = 0.0

        for c in range(self._actual_k_):
            mask = (labels == c) & (~outlier_mask)
            if np.any(mask):
                total += np.sum((X[mask] - centers[c]) ** 2)

        return float(total)

    def _find_weak_points(self, X, labels, centers, outlier_mask):
        """
        Détecte les points faibles de chaque agent-cluster.

        Un point faible est un point éloigné de son centre.
        On sélectionne un pourcentage des points les plus éloignés dans chaque cluster.
        """
        weak_indices = []

        for c in range(self._actual_k_):
            cluster_indices = np.where((labels == c) & (~outlier_mask))[0]

            if len(cluster_indices) <= self.min_cluster_size:
                continue

            Xc = X[cluster_indices]
            distances = np.sum((Xc - centers[c]) ** 2, axis=1)

            n_weak = max(1, int(np.ceil(self.weak_points_ratio * len(cluster_indices))))
            local_weak = np.argsort(distances)[-n_weak:]

            weak_indices.extend(cluster_indices[local_weak])

        return weak_indices

    def _try_transfer_point(self, X, labels, centers, outlier_mask, index):
        """
        Essaie de transférer un point faible vers un autre cluster.

        Le transfert est accepté seulement s'il réduit l'inertie globale.
        """
        current_cluster = labels[index]

        if outlier_mask[index]:
            return False

        # On évite de vider un cluster.
        if np.sum((labels == current_cluster) & (~outlier_mask)) <= self.min_cluster_size:
            return False

        old_inertia = self._compute_inertia(X, labels, centers, outlier_mask)

        x = X[index]
        distances = np.sum((centers - x) ** 2, axis=1)
        candidate_clusters = np.argsort(distances)

        for target_cluster in candidate_clusters:
            if target_cluster == current_cluster:
                continue

            new_labels = labels.copy()
            new_labels[index] = target_cluster

            new_centers = self._compute_centers(X, new_labels, outlier_mask)
            new_inertia = self._compute_inertia(X, new_labels, new_centers, outlier_mask)

            if new_inertia < old_inertia - self.exchange_tolerance:
                labels[index] = target_cluster
                return True

        return False

    def _detect_outliers(self, X, labels, centers, outlier_mask):
        """
        Marque comme outliers les points trop éloignés du centre de leur cluster.

        outlier_factor joue le rôle d'un seuil :
        plus il est grand, moins on détecte d'outliers.
        """
        changed = 0

        for c in range(self._actual_k_):
            indices = np.where((labels == c) & (~outlier_mask))[0]

            if len(indices) <= self.min_cluster_size:
                continue

            Xc = X[indices]
            distances = np.sqrt(np.sum((Xc - centers[c]) ** 2, axis=1))

            mean_d = np.mean(distances)
            std_d = np.std(distances)

            threshold = mean_d + self.outlier_factor * std_d

            outlier_local = indices[distances > threshold]

            for idx in outlier_local:
                # On évite de supprimer trop de points d'un cluster.
                if np.sum((labels == c) & (~outlier_mask)) > self.min_cluster_size:
                    outlier_mask[idx] = True
                    changed += 1

        return changed

    def fit(self, X, y=None):
        """
        Entraîne l'algorithme.

        Étapes :
        1) choisir k automatiquement avec G-Split si demandé ;
        2) initialiser K-means avec ce k ;
        3) appliquer la logique des agents :
            - recalcul des centres,
            - détection des points faibles,
            - transfert bénéfique,
            - détection d'outliers.
        """
        X = check_array(X, dtype=float)
        n_samples = X.shape[0]

        # ---------------------------------------------------------------------
        # Étape 1 : choix automatique de k
        # ---------------------------------------------------------------------
        if self.n_clusters == "auto_gsplit":
            best_k, gsplit_info = select_k_by_gsplit(
                X,
                k_initial=self.k_initial,
                k_max=self.k_max,
                min_cluster_size=self.min_cluster_size,
                min_inertia_gain=self.min_inertia_gain,
                min_silhouette_split=self.min_silhouette_split,
                min_gaussian_gap=self.min_gaussian_gap,
                random_state=self.random_state,
            )
            self._actual_k_ = int(best_k)
            self.gsplit_info_ = gsplit_info

        elif isinstance(self.n_clusters, int):
            self._actual_k_ = int(self.n_clusters)
            self.gsplit_info_ = {}

        else:
            raise ValueError("n_clusters doit être un entier ou 'auto_gsplit'.")

        if self._actual_k_ < 2:
            raise ValueError("Le nombre de clusters doit être >= 2.")

        if self._actual_k_ > n_samples:
            raise ValueError("Le nombre de clusters ne peut pas dépasser le nombre de points.")

        # ---------------------------------------------------------------------
        # Étape 2 : initialisation K-means classique avec le k trouvé
        # ---------------------------------------------------------------------
        km = KMeans(
            n_clusters=self._actual_k_,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=self.random_state,
        )

        labels = km.fit_predict(X)
        centers = km.cluster_centers_.copy()
        outlier_mask = np.zeros(n_samples, dtype=bool)

        self.history_ = {
            "inertia": [],
            "transfers": [],
            "outliers": [],
        }

        previous_inertia = np.inf

        # ---------------------------------------------------------------------
        # Étape 3 : boucle d'agents
        # ---------------------------------------------------------------------
        for iteration in range(1, self.max_iter + 1):

            centers = self._compute_centers(X, labels, outlier_mask)

            # Assignation standard au centre le plus proche pour les non-outliers.
            non_outliers = np.where(~outlier_mask)[0]
            distances = self._distances_squared(X[non_outliers], centers)
            labels[non_outliers] = np.argmin(distances, axis=1)

            centers = self._compute_centers(X, labels, outlier_mask)

            # Les agents choisissent leurs points faibles.
            weak_points = self._find_weak_points(X, labels, centers, outlier_mask)

            transfers = 0

            # Chaque point faible peut être transféré si cela améliore l'inertie.
            for idx in weak_points:
                moved = self._try_transfer_point(X, labels, centers, outlier_mask, idx)
                if moved:
                    transfers += 1
                    centers = self._compute_centers(X, labels, outlier_mask)

            # Détection des outliers.
            detected_outliers = self._detect_outliers(X, labels, centers, outlier_mask)
            centers = self._compute_centers(X, labels, outlier_mask)

            inertia = self._compute_inertia(X, labels, centers, outlier_mask)

            self.history_["inertia"].append(float(inertia))
            self.history_["transfers"].append(int(transfers))
            self.history_["outliers"].append(int(np.sum(outlier_mask)))

            improvement = previous_inertia - inertia

            # Critère d'arrêt :
            # si l'inertie ne s'améliore presque plus et qu'il n'y a plus de transferts.
            if improvement >= 0 and improvement < self.tol and transfers == 0 and detected_outliers == 0:
                break

            previous_inertia = inertia

        self.cluster_centers_ = centers
        self.labels_ = labels.copy()
        self.outlier_mask_ = outlier_mask.copy()
        self.final_labels_ = labels.copy()
        self.final_labels_[outlier_mask] = -1
        self.inertia_ = float(self.history_["inertia"][-1])
        self.n_iter_ = len(self.history_["inertia"])

        return self

    def predict(self, X):
        """
        Assigne de nouveaux points au centre le plus proche.
        """
        check_is_fitted(self, ["cluster_centers_"])
        X = check_array(X, dtype=float)
        distances = self._distances_squared(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, X, y=None):
        """
        Entraîne l'algorithme puis retourne les labels finaux.
        Label -1 = outlier.
        """
        self.fit(X, y)
        return self.final_labels_
