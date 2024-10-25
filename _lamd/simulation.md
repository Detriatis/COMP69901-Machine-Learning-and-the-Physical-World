---
week: 3
session: 1
title: "Simulation"
featured_image: slides/diagrams/simulation/Loafer.gif
abstract: >
  This lecture will introduce the notion of simulation and review the different types of simulation we might use to represent the physical world. 
author:
- family: Lawrence
  given: Neil D.
  gscholar: r3SJcvoAAAAJ
  institute: University of Cambridge
  twitter: lawrennd
  url: http://inverseprobability.com
layout: lecture
time: "12:00"
date: 2024-10-24
youtube: ieEFaGml4lM
ipynb: True
reveal: True
transition: None
---

\notes{Last lecture Carl Henrik introduced you to some of the challenges of approximate inference. Including the problem of mathematical tractability. Before that he introduced you to a particular form of model, the Gaussian process.}

\include{_software/includes/notutils-software.md}
\include{_software/includes/mlai-software.md}
\include{_simulation/includes/wolfram-automata.md}
\include{_simulation/includes/rule-30.md}
\include{_simulation/includes/game-of-life.md}
\include{_simulation/includes/wolfram-conway-life.md}
\include{_simulation/includes/packing-problems.md}

\include{_gp/includes/gp-intro-very-short.md}

\notes{So, Gaussian processes provide an example of a particular type of model. Or, scientifically, we can think of such a model as a mathematical representation of a hypothesis around data. The rejection sampling view of Bayesian inference can be seen as rejecting portions of that initial hypothesis that are inconsistent with the data. From a Popperian perspective, areas of the prior space are falsified by the data, leaving a posterior space that represents remaining plausible hypotheses.}

\notes{The flaw with this point of view is that the initial hypothesis space was also restricted. It only contained functions where the instantiated points from the function are jointly Gaussian distributed.}

\include{_gp/includes/planck-cmp-master-gp.md}
\include{_physics/includes/universe-gaussian.md}
\include{_physics/includes/precise-physical-laws.md}
\include{_physics/includes/simulation-scales.md}
\include{_ml/includes/accelerate-programme.md}
\include{_ml/includes/related-approaches.md}
\include{_ml/includes/probabilistic-programming.md}
\include{_ml/includes/approximate-bayesian-computation.md}
\include{_ml/includes/causality-section.md}
\include{_ml/includes/simulation-conclusion.md}

\thanks

\references
