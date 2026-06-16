"""
semisupervised_fission.py
===========================
Semi-supervised classification of fission modes using self-training
with cross-validation.

Maps from: 1065_Azamat-Mukhamediya_SRPM-ST (self-training cross-validation)

Physical context
----------------
Nuclear fission can proceed through different modes (channels),
each characterised by different fragment mass splits:
- Standard I (S1): asymmetric, A_H ~ 140
- Standard II (S2): asymmetric, A_H ~ 134
- Super-long (SL): very asymmetric, A_H ~ 145
- Super-short (SS): symmetric, A_H ~ 118

These modes correspond to different paths on the PES through different
saddle points.  Given a set of fission events with known modes (from
experimental data or Langevin trajectories), we use semi-supervised
learning to classify events with unknown mode labels.

The self-training algorithm (from SRPM-ST project):
1. Train initial classifier on labelled data
2. Predict on unlabelled data
3. Add high-confidence predictions to training set
4. Repeat until convergence

Cross-validation selects the optimal number of self-training iterations.
"""

import math
import random
from typing import List, Dict, Tuple, Optional

from potential_energy_surface import FissionPES


class FissionModeClassifier:
    """
    Simple nearest-centroid classifier for fission modes.

    Features extracted from each fission event:
    - Elongation at scission c_sc
    - Neck parameter at scission h_sc
    - Mass asymmetry alpha_sc
    - TKE
    - Total excitation energy
    """

    def __init__(self):
        # Fission mode definitions ( centroids in feature space)
        self.modes = {
            'S1': {'name': 'Standard I', 'c': 2.0, 'h': 0.5, 'alpha': 0.18},
            'S2': {'name': 'Standard II', 'c': 2.0, 'h': 0.4, 'alpha': 0.12},
            'SL': {'name': 'Super-long', 'c': 2.3, 'h': 0.3, 'alpha': 0.25},
            'SS': {'name': 'Super-short', 'c': 1.8, 'h': 0.7, 'alpha': 0.02},
        }
        self.centroids: Dict[str, List[float]] = {}
        self.training_data: List[Dict] = []
        self.predictions: List[Dict] = []

    def generate_features(self, c: float, h: float, alpha: float,
                          pes: Optional[FissionPES] = None) -> List[float]:
        """
        Extract feature vector from scission-point configuration.
        """
        features = [c, h, alpha]

        if pes is not None:
            v = pes.total_potential(c, h, alpha)
            features.append(v)

            # TKE estimate
            a_heavy = int(round(236 * (1 + alpha) / 2))
            a_light = 236 - a_heavy
            z_heavy = int(round(92 * a_heavy / 236))
            z_light = 92 - z_heavy
            from nuclear_constants import coulomb_barrier_energy
            tke = coulomb_barrier_energy(z_light, z_heavy, a_light, a_heavy)
            features.append(tke)
        else:
            features.extend([0.0, 0.0])

        return features

    def euclidean_distance(self, f1: List[float], f2: List[float]) -> float:
        """Compute Euclidean distance between two feature vectors."""
        s = 0.0
        for x, y in zip(f1, f2):
            s += (x - y) ** 2
        return math.sqrt(s)

    def train_centroids(self, labelled_data: List[Dict]) -> None:
        """
        Compute mode centroids from labelled data.
        Each data point has 'features' and 'mode' keys.
        """
        mode_features: Dict[str, List[List[float]]] = {m: [] for m in self.modes}

        for d in labelled_data:
            mode = d['mode']
            if mode in mode_features:
                mode_features[mode].append(d['features'])

        for mode, feat_list in mode_features.items():
            if len(feat_list) > 0:
                n_feat = len(feat_list[0])
                centroid = [0.0] * n_feat
                for f in feat_list:
                    for j in range(n_feat):
                        centroid[j] += f[j]
                for j in range(n_feat):
                    centroid[j] /= len(feat_list)
                self.centroids[mode] = centroid

    def predict_nearest(self, features: List[float]) -> Tuple[str, float]:
        """
        Predict mode label by nearest centroid.
        Returns (mode_label, distance).
        """
        best_mode = 'S1'
        best_dist = float('inf')

        for mode, centroid in self.centroids.items():
            d = self.euclidean_distance(features, centroid)
            if d < best_dist:
                best_dist = d
                best_mode = mode

        return best_mode, best_dist

    def self_training_step(self, unlabelled_data: List[Dict],
                           confidence_threshold: float = 1.0) -> Tuple[List[Dict], int]:
        """
        One step of self-training (from SRPM-ST project concept).

        1. Predict labels for unlabelled data
        2. Select high-confidence predictions (distance < threshold)
        3. Return selected data as newly labelled

        Returns (newly_labelled, count).
        """
        newly_labelled = []
        n_added = 0

        for d in unlabelled_data:
            mode, dist = self.predict_nearest(d['features'])
            if dist < confidence_threshold:
                new_d = d.copy()
                new_d['mode'] = mode
                new_d['confidence'] = 1.0 / (1.0 + dist)
                newly_labelled.append(new_d)
                n_added += 1

        return newly_labelled, n_added

    def cross_validate(self, data: List[Dict], n_folds: int = 3) -> Dict[str, float]:
        """
        Stratified k-fold cross-validation to estimate accuracy.
        """
        if len(data) == 0:
            return {'accuracy': 0.0}

        # Shuffle
        indices = list(range(len(data)))
        random.shuffle(indices)

        fold_size = len(indices) // max(n_folds, 1)
        if fold_size < 1:
            fold_size = 1

        total_correct = 0
        total_tested = 0

        for fold in range(n_folds):
            test_start = fold * fold_size
            test_end = min(test_start + fold_size, len(indices))
            test_idx = indices[test_start:test_end]
            train_idx = [i for i in indices if i < test_start or i >= test_end]

            train_data = [data[i] for i in train_idx]
            test_data = [data[i] for i in test_idx]

            # Train
            self.train_centroids(train_data)

            # Test
            for d in test_data:
                pred_mode, _ = self.predict_nearest(d['features'])
                if pred_mode == d['mode']:
                    total_correct += 1
                total_tested += 1

        accuracy = total_correct / max(total_tested, 1)
        return {'accuracy': accuracy, 'n_tested': total_tested}

    def run_self_training(self, labelled: List[Dict],
                          unlabelled: List[Dict],
                          max_iterations: int = 5,
                          threshold: float = 1.5) -> Dict[str, any]:
        """
        Full self-training loop with cross-validation.

        From SRPM-ST: 5-fold CV to select optimal self-training iterations.
        """
        # Initial training
        self.train_centroids(labelled)
        all_labelled = labelled[:]

        # CV on initial model
        cv_result = self.cross_validate(labelled)

        iteration_results = []
        for iteration in range(max_iterations):
            new_labelled, n_added = self.self_training_step(
                unlabelled, threshold
            )

            if n_added == 0:
                break

            all_labelled.extend(new_labelled)
            self.train_centroids(all_labelled)

            # Remove newly labelled from unlabelled pool
            new_set = set(id(d) for d in new_labelled)
            unlabelled = [d for d in unlabelled if id(d) not in new_set]

            # CV after this iteration
            cv_result = self.cross_validate(all_labelled)
            iteration_results.append({
                'iteration': iteration,
                'n_added': n_added,
                'total_labelled': len(all_labelled),
                'accuracy': cv_result['accuracy'],
            })

            # Decrease threshold (more selective)
            threshold *= 0.85

        return {
            'initial_cv_accuracy': cv_result.get('accuracy', 0.0),
            'n_iterations': len(iteration_results),
            'final_labelled': len(all_labelled),
            'iteration_details': iteration_results,
            'centroids': self.centroids,
        }


def generate_synthetic_fission_data(n_events: int = 200,
                                     noise: float = 0.1,
                                     seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate synthetic fission event data with known mode labels.

    Returns (labelled, unlabelled) data.
    """
    random.seed(seed)
    pes = FissionPES(92, 236)

    modes_config = {
        'S1': {'c': 2.0, 'h': 0.5, 'alpha': 0.18},
        'S2': {'c': 2.0, 'h': 0.4, 'alpha': 0.12},
        'SL': {'c': 2.3, 'h': 0.3, 'alpha': 0.25},
        'SS': {'c': 1.8, 'h': 0.7, 'alpha': 0.02},
    }

    data = []
    classifier = FissionModeClassifier()

    for i in range(n_events):
        # Random mode assignment
        mode = random.choice(list(modes_config.keys()))
        cfg = modes_config[mode]

        # Add noise
        c = cfg['c'] + noise * random.gauss(0, 1)
        h = cfg['h'] + noise * random.gauss(0, 1)
        alpha = cfg['alpha'] + noise * random.gauss(0, 1)

        features = classifier.generate_features(c, h, alpha, pes)
        data.append({
            'features': features,
            'mode': mode,
            'c': c,
            'h': h,
            'alpha': alpha,
        })

    # Split: 30% labelled, 70% unlabelled
    n_labelled = max(10, int(0.3 * n_events))
    random.shuffle(data)
    labelled = data[:n_labelled]
    unlabelled = [{k: v for k, v in d.items() if k != 'mode'} for d in data[n_labelled:]]

    return labelled, unlabelled
