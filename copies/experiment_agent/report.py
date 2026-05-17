# file: finance_kg_experiment/report.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from ontology import OntologyCandidate
from modeling import ModelMetrics
import json


@dataclass
class CandidateResult:
    candidate: OntologyCandidate
    metrics: ModelMetrics
    step_index: int
    candidate_index: int


@dataclass
class ExperimentReport:
    results: List[CandidateResult]

    def best_by_auc(self) -> CandidateResult:
        return max(self.results, key=lambda r: r.metrics.auc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [
                {
                    "candidate_name": r.candidate.name,
                    "description": r.candidate.description,
                    "step_index": r.step_index,
                    "candidate_index": r.candidate_index,
                    "auc": r.metrics.auc,
                    "f1": r.metrics.f1,
                    "n_train": r.metrics.n_train,
                    "n_val": r.metrics.n_val,
                }
                for r in self.results
            ]
        }

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def pretty_print(self):
        print("\n=== Ontology Experiment Report ===")
        print(f"Total candidates tested: {len(self.results)}")
        for r in self.results:
            print(
                f"\nStep {r.step_index} candidate {r.candidate_index}: {r.candidate.name}\n"
                f"  Desc: {r.candidate.description}\n"
                f"  AUC: {r.metrics.auc:.4f} | F1: {r.metrics.f1:.4f}\n"
                f"  Train N: {r.metrics.n_train} | Val N: {r.metrics.n_val}"
            )
        best = self.best_by_auc()
        print("\nBest candidate by AUC:")
        print(f"  {best.candidate.name} (step {best.step_index}, idx {best.candidate_index})")
        print(f"  AUC: {best.metrics.auc:.4f} | F1: {best.metrics.f1:.4f}")

