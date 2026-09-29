import unittest
import numpy as np
from dbec import DBEC, entropy, embeddings_within_epsilon, mean_nn_distance

class TestDBEC(unittest.TestCase):
    def setUp(self):
        # Create some dummy data (avoid all zeros to prevent cosine similarity issues)
        self.embeddings = np.array([
            [0.1, 0.1],
            [0.2, 0.2],
            [0.1, 0.2],
            [1.0, 1.0],
            [1.1, 1.1],
            [1.0, 1.1],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.0, 5.1],
            [5.1, 5.0]
        ])
        
        self.labels = ['A', 'A', 'B', 'C', 'C', 'C', 'D', 'E', 'F', 'G']

    def test_entropy(self):
        # High entropy: all different labels
        ent_high = entropy([6, 7, 8, 9], self.labels, min_points=3)
        self.assertGreater(ent_high, 0)
        
        # Low/Zero entropy: all same labels
        ent_low = entropy([3, 4, 5], self.labels, min_points=3)
        self.assertEqual(ent_low, 0)

        # Min points constraint
        ent_min_points = entropy([0, 1], self.labels, min_points=3)
        self.assertEqual(ent_min_points, -1)

    def test_embeddings_within_epsilon(self):
        # Euclidean Test
        eps_euclidean = 0.2
        neighbors_euclidean = embeddings_within_epsilon(self.embeddings, self.embeddings[0], eps_euclidean, metric='euclidean')
        self.assertGreater(len(neighbors_euclidean), 0)

        # Cosine Test
        # Points on same line [1.0, 1.0] and [5.0, 5.0] have 0 cosine distance
        eps_cosine = 0.01
        neighbors_cosine = embeddings_within_epsilon(self.embeddings, self.embeddings[3], eps_cosine, metric='cosine')
        self.assertGreater(len(neighbors_cosine), 0)

    def test_mean_nn_distance(self):
        # Euclidean Test
        dist_euclidean = mean_nn_distance(self.embeddings, neighbor=2, metric='euclidean')
        self.assertIsInstance(dist_euclidean, float)

        # Cosine Test
        dist_cosine = mean_nn_distance(self.embeddings, neighbor=2, metric='cosine')
        self.assertIsInstance(dist_cosine, float)

    def test_dbec_model(self):
        for metric in ['euclidean', 'cosine']:
            with self.subTest(metric=metric):
                model = DBEC(neighbor=2, min_points_entropy=2, min_samples_high=0.1, entropy_threshold=0.1, metric=metric)
                labels_ = model.fit_predict(self.embeddings, self.labels)
                
                self.assertEqual(len(labels_), len(self.embeddings))
                self.assertIsInstance(labels_, np.ndarray)

    def test_dbec_high_dimensional(self):
        # Create 100 random embeddings of dimension 384 (like SentenceTransformers)
        # We ensure they are positive to avoid zero vectors for cosine
        high_dim_embeddings = np.random.rand(100, 384) + 0.1 
        
        # Create random labels from 4 categories
        high_dim_labels = np.random.choice(['Topic A', 'Topic B', 'Topic C', 'Topic D'], size=100)
        
        for metric in ['euclidean', 'cosine']:
            with self.subTest(metric=metric):
                model = DBEC(neighbor=5, min_points_entropy=5, min_samples_high=0.3, entropy_threshold=0.3, metric=metric)
                labels_ = model.fit_predict(high_dim_embeddings, high_dim_labels)
                
                self.assertEqual(len(labels_), 100)
                self.assertIsInstance(labels_, np.ndarray)

    def test_run_metrics(self):
        # Verify get_run_metrics captures and returns the expected dictionary
        model = DBEC(neighbor=2, min_points_entropy=2, min_samples_high=0.1, entropy_threshold=0.1)
        
        # Test calling get_run_metrics
        metrics = model.get_run_metrics(self.embeddings, self.labels)
        
        # Check that it returns a dictionary
        self.assertIsInstance(metrics, dict)
        
        # Check that essential keys are present in the response
        expected_keys = [
            'dataset_labels', 'max_entropy', 'entropy_boundary', 
            'number_low', 'number_high', 'number_core', 
            'clusters', 'num_clusters', 'point_entropy', 'point_type', 
            'entropy_clusters', 'cluster_sizes', 'cluster_c_counts', 
            'cluster_b_counts', 'cluster_h_counts', 'cluster_l_counts', 
            'cluster_o_counts'
        ]
        
        for key in expected_keys:
            self.assertIn(key, metrics)
            
        # Basic logical checks
        self.assertGreaterEqual(metrics['num_clusters'], 0)
        self.assertGreaterEqual(metrics['max_entropy'], 0)

if __name__ == '__main__':
    unittest.main()
