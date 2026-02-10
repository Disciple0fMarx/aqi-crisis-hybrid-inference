"""
MODULE: Fuzzy Association Rule Mining (F-Apriori)
PURPOSE: Extracts high-confidence 'Knowledge Rules' from fuzzified air quality data.
METHODOLOGY:
    - Fuzzy Support: Calculates the frequency of itemsets by summing membership degrees 
      rather than binary counts, preserving uncertainty.
    - Pattern Discovery: Identifies relationships between secondary pollutants 
      (e.g., NO2, CO) and PM2.5 Hazardous events.
    - Rule Injection: Prepares discovered logic for integration as 'expert features' 
      into the LSTM forecasting model.
"""

import pandas as pd
import numpy as np
from itertools import combinations
from os import path, makedirs
import json


class FuzzyApriori:
    def __init__(self, min_support=0.05, min_confidence=0.6):
        self.min_support = min_support
        self.min_confidence = min_confidence

    def calculate_1_itemsets(self, df, fuzzy_cols):
        """Finds support for single linguistic states."""
        n = len(df)
        frequent_1 = {}
        for col in fuzzy_cols:
            support = df[col].sum() / n
            if support >= self.min_support:
                frequent_1[col] = support
        return frequent_1

    def calculate_2_itemsets(self, df, frequent_1):
        """
        Joins frequent 1-itemsets to find pairs.
        Uses the 'min' operator for Fuzzy Intersection (T-norm).
        """
        n = len(df)
        frequent_2 = {}
        items = list(frequent_1.keys())
        
        # Generate all possible pairs
        for item_a, item_b in combinations(items, 2):
            # Optimization: Don't pair states of the same pollutant (e.g., PM2.5_Good & PM2.5_Hazard)
            if item_a.split('_')[0] == item_b.split('_')[0]:
                continue
                
            # Fuzzy AND: Intersection is the minimum of the two memberships
            combined_membership = np.minimum(df[item_a], df[item_b])
            support = combined_membership.sum() / n
            
            if support >= self.min_support:
                frequent_2[(item_a, item_b)] = support
                
        return frequent_2

    def extract_rules(self, frequent_1, frequent_2, target_item="PM2.5_Hazardous"):
        """
        Finds rules of the form: {Pollutant A} -> {Target}
        Confidence = Support(A and Target) / Support(A)
        """
        rules = []
        for (item_a, item_b), support_ab in frequent_2.items():
            # We specifically look for rules leading to our target
            if item_b == target_item:
                antecedent = item_a
            elif item_a == target_item:
                antecedent = item_b
            else:
                continue
            
            confidence = support_ab / frequent_1[antecedent]
            if confidence >= self.min_confidence:
                rules.append({
                    "rule": f"IF {antecedent} THEN {target_item}",
                    "support": support_ab,
                    "confidence": confidence
                })
        return rules


def save_rules(rules, filename="rules.json"):
    """Saves discovered fuzzy rules to the metadata directory."""
    output_path = path.join("metadata", filename)
    
    # Ensure metadata directory exists
    if not path.exists("metadata"):
        makedirs("metadata")
        
    with open(output_path, 'w') as f:
        json.dump(rules, f, indent=4)
    print(f"✓ {len(rules)} Knowledge Rules exported to: {output_path}")


def run_mining_session(city_name='Delhi', filename="rules.json"):
    df = pd.read_csv(f"data/processed/fuzzified_{'_'.join(city_name.lower().split(' '))}_data.csv")
    
    # Filter for all fuzzy membership columns
    fuzzy_cols = [c for c in df.columns if any(s in c for s in ['_Good', '_Moderate', '_Hazardous'])]
    
    engine = FuzzyApriori(min_support=0.03, min_confidence=0.4)
    
    print("Step 1: Mining Frequent 1-Itemsets...")
    f1 = engine.calculate_1_itemsets(df, fuzzy_cols)
    
    print(f"Step 2: Mining Frequent 2-Itemsets from {len(f1)} candidates...")
    f2 = engine.calculate_2_itemsets(df, f1)
    
    print("Step 3: Extracting Knowledge Rules for PM2.5 Crisis...")
    rules = engine.extract_rules(f1, f2, target_item="PM2.5_Hazardous")
    
    print("\n--- Discovered Knowledge Rules ---")
    for r in rules:
        print(f"{r['rule']} | Conf: {r['confidence']:.2f} | Sup: {r['support']:.2f}")
        
    if rules:
        save_rules(rules, filename)
    else:
        print("[!] No rules found. Try lowering min_support or min_confidence.")


if __name__ == "__main__":
    run_mining_session()
