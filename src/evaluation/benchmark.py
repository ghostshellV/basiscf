import pandas as pd
import torch
from src.evaluation.metrics import validity_score, proximity_score, smoothness_score

class BenchmarkRunner:
    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader
        self.explainers = {} # e.g., {'TiBa': tiba_explainer, 'CoMTE': comte_explainer}

    def register_explainer(self, name, explainer_instance):
        self.explainers[name] = explainer_instance

    def run(self, num_samples=50):
        results = []
        # Loop over a subset of test data
        for i, (x, y_true) in enumerate(self.test_loader):
            if i >= num_samples: break
            
            target_rul = y_true + 20 # Example: Try to increase RUL by 20

            for name, explainer in self.explainers.items():
                # Generate CF
                cf = explainer.generate(x, target_rul)
                
                # Compute Metrics
                res = {
                    'Method': name,
                    'Validity': validity_score(self.model, cf, target_rul),
                    'Proximity': proximity_score(x, cf),
                    'Smoothness': smoothness_score(cf) # Your method should win here
                }
                results.append(res)
        
        return pd.DataFrame(results)