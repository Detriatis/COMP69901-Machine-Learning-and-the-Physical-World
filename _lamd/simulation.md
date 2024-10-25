---
week: 3
session: 1
title: "Simulation"
featured_image: slides/diagrams/simulation/Loafer.gif
abstract:  >
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
oldyoutube: 
- code: ieEFaGml4lM
  year: 2022
- code: k46Av3liq7M
  year: 2021
- code: AmI5nq8s4qc
  year: 2020
ipynb: True
reveal: True
transition: None
---


\notes{Last lecture Carl Henrik introduced you to some of the challenges of approximate inference. Including the problem of mathematical tractability. Before that he introduced you to a particular form of model, the Gaussian process.}

\include{_notebooks/includes/plot-setup.md}
\include{_software/includes/notutils-software.md}
\include{_software/includes/mlai-software.md}

\include{_books/includes/the-structure-of-scientific-revolutions.md}

\include{_simulation/includes/wolfram-automata.md}
\include{_simulation/includes/rule-30.md}
\include{_simulation/includes/game-of-life.md}
\include{_simulation/includes/wolfram-conway-life.md}
\include{_simulation/includes/packing-problems.md}


\include{_gp/includes/gp-intro-very-short.md}

\notes{So, Gaussian processes provide an example of a particular type of model. Or, scientifically, we can think of such a model as a mathematical representation of a hypothesis around data. The rejection sampling view of Bayesian inference can be seen as rejecting portions of that initial hypothesis that are inconsistent with the data. From a Popperian perspective, areas of the prior space are falsified by the data, leaving a posterior space that represents remaining plausible hypotheses.}

\notes{The flaw with this point of view is that the initial hypothesis space was also restricted. It only contained functions where the instantiated points from the function are jointly Gaussian distributed.}

\include{_gp/includes/planck-cmp-master-gp.md}

\notes{Those cosmological simulations are based on a relatively simple set of 'rules' that stem from our understanding of natural laws. These 'rules' are mathematical abstractions of the physical world. Representations of behavior in mathematical form that capture the interaction forces between particles. The grand aim of physics has been to unify these rules into a single unifying theory. Popular understanding of this quest developed because of Stephen Hawking's book, "[A Brief History of Time](https://en.wikipedia.org/wiki/A_Brief_History_of_Time)". The idea of these laws as 'ultimate causes' has given them a pseudo religious feel, see for example Paul Davies's book "[The Mind of God](https://en.wikipedia.org/wiki/The_Mind_of_God)" which comes from a quotation form Stephen Hawking. }

\newslide{}

> If we do discover a theory of everything ... it would be the ultimate triumph of human reason-for then we would truly know the mind of God
>
> Stephen Hawking in *A Brief History of Time* 1988

\speakernotes{Nice quote but having a unifying theory doesn't give us omniscience.}

\notes{This is an entrancing quote, that seems to work well for selling books (A Brief History of Time sold over 10 million copies), but as Laplace has already pointed out to us, the Universe doesn't work quite so simply as that. Commonly, God is thought to be omniscient, but having a grand unifying theory alone doesn't give us omniscience.}

\notes{Laplace's demon still applies. Even if we had a grand unifying theory, which encoded "all the forces that set nature in motion" we have an amount of work left to do in any quest for 'omniscience'.}

\newslide{}

> We may regard the present state of the universe as the effect of its
> past and the cause of its future. An intellect which at a certain
> moment would know all forces that set nature in motion, and all
> positions of all items of which nature is composed, ...
\newslide{}
> ... if this intellect
> were also vast enough to submit these data to analysis, it would
> embrace in a single formula the movements of the greatest bodies of
> the universe and those of the tiniest atom; for such an intellect
> nothing would be uncertain and the future just like the past would be
> present before its eyes.
>
> --- Pierre Simon Laplace [@Laplace-essai14]

\speakernotes{Laplace's demon requires us to also know positions of all items and to submit the data to analysis.}

\newslide{}

\notes{We summarized this notion as }
$$
\text{data} + \text{model} \stackrel{\text{compute}}{\rightarrow} \text{prediction}
$$
\notes{As we pointed out, there is an irony in Laplace's demon forming the cornerstone of a movement known as 'determinism', because Laplace wrote about this idea in an essay on probabilities. The more important quote in the essay was }

\include{_physics/includes/laplaces-gremlin.md}

\newslide{}

\notes{Carl Henrik described how a prior probability $p(\parameterVector)$ represents our hypothesis about the way the world might behave. This can be combined with a *likelihood* through the process of multiplication. Correctly normalized, this gives an updated hypothesis that represents our *posterior* belief about the model in the light of the data.

There is a nice symmetry between this approach and how Karl Popper describes the process of scientific discovery. In *Conjectures and Refutations* (@Popper:conjectures63), Popper describes the process of scientific discovery as involving hypothesis and experiment. In our description hypothesis maps onto the *model*. The model is an abstraction of the hypothesis, represented for example as a set of mathematical equations, a computational description, or an analogous system (physical system). The data is the product of previous experiments, our readings, our observation of the world around us. We can combine these to make a prediction about what we might expect the future to hold. Popper's view on the philosophy of science was that the prediction should be falsifiable. 

We can see this process as a spiral driving forward, importantly Popper relates the relationship between hypothesis (model) and experiment (predictions) as akin to the relationship between the chicken and the egg. Which comes first? The answer is that they co-evolve together.}

\include{_data-science/includes/experiment-analyze-design-diagram.md}

\newslide{}

\figure{\includediagram{\diagramsDir/physics/different-models}{90%}}{The sets of different models. There are all the models in the Universe we might like to work with. Then there are those models that are computable e.g., by a Turing machine. Then there are those which are analytical tractable. I.e., where the solution might be found analytically. Finally, there are Gaussian processes, where the joint distribution of the states in the model is Gaussian.}


\notes{The approach we've taken to the model so far has been severely limiting. By constraining ourselves to models for which the mathematics of probability is tractable, we severely limit what we can say about the universe.

Although Bayes' rule only implies multiplication of probabilities, to acquire the posterior we also need to normalize. Very often it is this normalization step that gets in the way. The normalization step involves integration over the updated hypothesis space, to ensure the updated posterior prediction is correct.

We can map the process of Bayesian inference onto the $\text{model} + \text{data}$ perspective in the following way. We can see the model as the prior, the data as the likelihood and the prediction as the posterior[^mapping]. 

[^mapping]: We should be careful about such mappings, this is the one I prefer to think about because I try to think of my modelling assumptions as being stored in a probabilistic model, which I see as the prior distribution over what I expect the data to look like. In many domains of parametric modelling, however, the prior will be specified over the parameters of a model. In the Gaussian process formalism we're using, this mapping is clearer though. The 'prior' is the Gaussian process prior over functions, the data is the relationship between those functions and observations we make. This mental model will also suit what follows in terms of our consideration of simulation. But it would likely confuse someone who had only come to Bayesian inference through parametric models such a neural network. Note that even in such models, there will be a way of writing down the decomposition of the model that is akin to the above, but it might involve writing down intractable densities, so it's often avoided.}

\newslide{}

\notes{So, if we think of our model as incorporating what we know about the physical problem of interest (from Newton, or Bernoulli or Laplace or Einstein or whoever) and the data as being the observations (e.g., from Piazzi's telescope or a particle accelerator) then we can make predictions about what we might expect to happen in the future by combining the two. It is *those* predictions that Popper sees as important in verifying the scientific theory (which is incorporated in the model).

But while Gaussian processes are highly flexible non-parametric function models, they are *not* going to be sufficient to capture the type of physical processes we might expect to encounter in the real world. To give a sense, let's consider a few examples of the phenomena we might want to capture, either in the scientific world, or in real world decision making.}

\section{Precise Physical Laws}

\slides{* Newton's laws
* Huygens and conservation of energy
* Daniel Bernoulli and the kinetic theory of gases
* Modern climate simulation and Navier-Stokes equations
}

\speakernotes{Precise physical laws are predictive of the future. Met office super computer uses 1 km grids cells to compute the weather.}

\notes{We've already reviewed the importance of Newton's laws in forging our view of science: we mentioned the influence [Christiaan Huygens'](https://en.wikipedia.org/wiki/Christiaan_Huygens) work on collisions had on Daniel Bernoulli in forming the kinetic theory of gases. These ideas inform many of the physical models we have today around a number of natural phenomena. The MET Office supercomputer in Exeter spends its mornings computing the weather across the world its afternoons modelling climate scenarios. It uses the same set of principles that Newton described, and Bernoulli explored for gases. They are encoded in the Navier-Stokes equations. Differential equations that govern the flow of compressible and incompressible fluids. As well as predicting our weather, these equations are used in fluid dynamics models to understand the flight of aircraft, the driving characteristics of racing cars and the efficiency of gas turbine engines.

This broad class of physical models, or 'natural laws' is probably the closest to what Laplace was referring to in the demon. The search for unifying physical laws that dictate everything we observe around us has gone on. Alongside Newton we must mention James Clerk Maxwell, who unified electricity and magnetism in one set of equations that were inspired by the work and ideas of Michael Faraday. And still today we look for unifying equations that bring together in a single mathematical model the 'natural laws' we observe. One equation that for Laplace would be "all forces that set nature in motion". We can think of this as our first time of physical model, a 'precise model' of the known laws of our Universe, a model where we expect that the mapping from the mathematical abstraction to the physical reality is 'exact'.[^exact]

[^exact]: Unfortunately, I have to use the term 'exact' loosely here! For example, most of these laws treat space/time as a continuum. But in reality, it is quantised. The smallest length we can define is Planck length ($1.61 \times 10^{-35}$), and the the smallest time is Planck time. So even in this exact world of Maxwell and Newton there is an abstraction.}

\newslide{}

\include{_physics/includes/simulation-scales.md}

\subsection{How Machine Learning Can Help}

\notes{Machine learning models can often capture some of the regularities of the system that we find in these mergent properties. They do so, not from first principles, but from analysis of the data. In the Atomic Human, I argue that this has more in common with how human intellience solves problems than through first-principles modelling. When it comes to ML and the PHysical World, the aim is to use machine learning models alongside simulations to get the best of both worlds.}

\addatomic{}

\reading

\thanks

\references

