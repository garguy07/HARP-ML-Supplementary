# HARP-ML Evaluation Repositories

This directory lists the 14 real-world repositories from the NICHE dataset used to evaluate HARP-ML on the real-world benchmark.

## Repository Selection Criteria

Repositories were selected from the NICHE dataset based on the following criteria, as described in the paper:

- Labelled as engineered projects in the NICHE dataset
- Medium-sized (1,000 to 10,000 LOC)
- At least 100 GitHub stars
- At least 100 commits
- Organised into modules reflecting a well-defined component structure

All repositories are publicly available on GitHub.

CodeSmile was then executed on this candidate pool to identify repositories containing instances of the four targeted smells.

## File Description

`niche_repositories.csv` contains the following columns:

| Column          | Description                                            |
| --------------- | ------------------------------------------------------ |
| repository_name | Name of the repository                                 |
| github_url      | URL to the repository on GitHub                        |
| domain          | Application domain of the repository                   |
| stars_approx    | Approximate GitHub star count at time of evaluation    |
| selected_smells | Which targeted smells were detected in this repository |

## References

Widyasari, R. et al. (2023). NICHE: A Curated Dataset of Engineered Machine Learning Projects in Python. In Proceedings of the 20th International Conference on Mining Software Repositories (MSR 2023). https://doi.org/10.1109/MSR59073.2023.00022

Gilberto Recupito, Giammaria Giordano, Filomena Ferrucci, Dario Di Nucci, and Fabio Palomba. 2025. When code smells meet ML: on the lifecycle of ML-specific code smells in ML-enabled systems. Empirical Softw. Engg. 30, 5 (May 2025). https://doi.org/10.1007/s10664-025-10676-4
