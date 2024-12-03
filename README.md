# EqualizedCoverage_CP

This is a repo for the paper titled 'Bridging Fairness and Uncertainty: Theoretical Insights and
Practical Strategies for Equalized Coverage in GNNs'

## Introduction

Graph Neural Networks (GNNs) have been extensively used in many domains, such as social network analysis, financial fraud detection, and drug discovery. Prior research primarily concentrated on improving prediction accuracy while overlooking how reliable the model predictions are. Conformal prediction on graphs offers a promising solution to offer statistically sound uncertainty estimates with a pre-defined coverage level. Despite the promising progress, existing works only focus on achieving model coverage guarantees without considering fairness in the coverage within different demographic groups. To bridge the gap between conformal prediction and fair coverage across different groups, we pose the question: \emph{How can we enable the uncertainty estimates to be fairly applied across demographic groups?} To answer this question, for the first time, we propose a theoretical framework demonstrating that fair GNNs can enforce the same uncertainty bounds across different demographic groups, thereby minimizing bias in uncertainty estimates. Inspired by the theoretical results, we conduct comprehensive experiments across multiple fair GNN models to identify and analyze the key strategies that contribute to ensuring equalized uncertainty estimates. This extensive experimental analysis validates our theoretical findings and sheds light on the practical implications and potential adjustments needed to enhance fairness in GNN applications.

## Requirement

To install requirements, run:

    pip install -r requirements.txt

## Example to run

    python examples/model_name.py 
  
For example, if we want to run Fairwalk, we can execute the following command:

    python examples/fairwalk.py
