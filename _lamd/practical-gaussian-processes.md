---
week: 2
session: 2
title: "Practical Gaussian Processes"
abstract:  >
  This lecture will cover the practical side to Gaussian processes.
layout: lecture
author:
- given: Carl Henrik
  family: Ek
  institution: University of Cambridge
  url: http://carlhenrik.com
youtube: 
time: "12:00"
featured_image: 
pdfslides: l48-mlpw-04.pdf
pdfworksheet: practical-gaussian-processes.pdf
date: 2024-10-22
featured_image: slides/diagrams/gp/learning-a-manifold-of-fonts.png
transition: None
ipynb: true
abstract: |
  Gaussian processes provide a probability measure that allows us to perform statistical inference over the space of functions. While GPs are nice as mathematical objects when we need to implement them in practice we often run into issues. In this worksheet we will do a little bit of a whirlwind tour of a couple of approaches to address these problems. We will look at how we can address the numerical issues that often appear and we will look at approximations to circumvent the computational cost associated with Gaussian processes. Importantly when continuing using these models in the course you are most likely not going to implement them yourself but instead use some of the many excellent software packages that exists. The methods that we describe here are going to show you how these packages implement GPs and it will hopefully give you an idea of the type of thinking that goes into implementation of machine learning models.
---



\include{_notebooks/includes/notebook-setup.md}

\notes{So far we have not done any type of learning with Gaussian processes. Learning is the process where we adapt the parameters of the model to the data that we observe. For a probabilistic model the object here is to fit the distribution of the model such that the data has high likelihood under the model. You can do this at many different levels, you can fit the parameters of the likelihood directly to data, referred to as maximum likelihood but in that case you have not taken into account the information in the prior, i.e. you are fitting the data in a completely unconstrained way which is likely to lead to overfitting. Instead we try to marginalise out as many of the parameters as we can to reflect the knowledge that we have of the problem and then optimise the remaining ones.}

\notes{Remember, there will always be parameters left in a model as long as you place a parametrised distribution that you integrate out. For Gaussian processes, in most practical applications, we marginalise out the prior over the function values, $p(f | \theta)$ from the likelihood to reach the marginal likelihood,}

$$p(y | X, \theta) = \int p(y | f)p(f | X, \theta)d\theta$$

\notes{to reach the marginal likelihood which does not have $f$ as parameters. However, this distribution still has the parameters of the prior $\theta$ as a dependency.[^parameters] Learning now implies altering these parameters so that we find the ones that the probability of the observed data is maximised,}

[^parameters]: sometimes you will hear these referred to as hyper-parameters.

$$\theta^* = \argmax_\theta p(y | X, \theta)$$

\notes{Now let us do this specifically for a zero mean Gaussian process prior. We will focus on the zero mean setting as it is usually not the mean that gives us problems but the covariance function. However, if you want to have a parametrised mean function most of the things that we talk about will be the same. Given that most distributions you will ever work with is in the exponential class the normal approach to this is to maximise the log marginal likelihood. If we write this up for a Gaussian process it will have the following form,}

$$\log p(y | X, \theta) = \int p(y | f)p(f | X, \theta)d\theta$$
$$ = -\frac{1}{2}y^T\left(k(X, X + \beta^{-1}I)\right)^{-1}y - \frac{1}{2}\log \det\left(k(X, X) + \beta^{-1}I\right) - \frac{N}{2}\log 2\pi$$

\section{Numerical Stability}

\notes{To address the numerical issues related to the two expressions above we are going to exploit that the problem that we work on is actually quite structured. The covariance matrix is in a class of matrices that are called positive-definite matrices meaning they are symmetric, full rank and with all eigen-values positive. This we can exploit to make computations better conditioned. To do so we are going to use the Cholesky decomposition which allows us to write a positive definite matrix as a product of a lower-triangular matrix and its transpose,}

$$K = LL^T$$

\notes{Let's first look at how the decomposition can be used to compute the log determinant term in the marginal log-likelihood,}

\pythonblockstart
import numpy as np
from scipy import linalg

def compute_logdet(K):
    """
    Compute log determinant of matrix K using Cholesky decomposition
    
    Parameters:
    K (ndarray): Positive definite matrix
    
    Returns:
    float: log determinant of K
    """
    L = np.linalg.cholesky(K)
    return 2 * np.sum(np.log(np.diag(L)))
\pythonblockend

\notes{The decomposition allows us to write:}

$$\log \det K = \log \det(LL^T) = \log (\det L)^2$$

\notes{Now we have to compute the determinant of the Cholesky factor $L$, this turns out to be very easy as the determinant of a upper/lower diagonal matrix is the product of the values on the diagonal,}

$$\det L = \prod_{i} \ell_{ii}$$

\notes{If we put everything together we get:}

$$\log \det K = \log\left(\prod_{i} \ell_{ii}\right)^2 = 2\sum_{i} \ell_{ii}$$

\notes{So the log determinant of the covariance matrix is simply the sum of the diagonal elements of the Cholesky factor. From our first classes in Computer science we know that summing lots of values of different scale is much better for keeping precision compared to taking the product.}

\notes{You can also use the Cholesky factors in order to address the term that includes the inverse and making this better conditioned. What we will do is to solve the inverse by solving two systems of linear equations, both who have already been made into upper and lower triangular form therefore being trivial to solve.}

\pythonblockstart
def solve_cholesky(K, y):
    """
    Solve system Kx = y using Cholesky decomposition
    
    Parameters:
    K (ndarray): Positive definite matrix
    y (ndarray): Right hand side vector
    
    Returns:
    ndarray: Solution x
    """
    L = np.linalg.cholesky(K)
    # Forward substitution
    z = linalg.solve_triangular(L, y, lower=True)
    # Backward substitution 
    x = linalg.solve_triangular(L.T, z, lower=False)
    return x
\pythonblockend

\notes{Let's write down how this works step by step. For solving the system $Ax = b$, instead of computing $x = A^{-1}b$ directly, we use the Cholesky decomposition $A = LL^T$ giving us:}

$$LL^Tx = b$$

\notes{We can solve this in two steps:}

1. First solve $Lz = b$ by forward substitution:
   $$\begin{aligned}
   \ell_{1,1}z_1 &= b_1 \\
   \ell_{2,1}z_1 + \ell_{2,2}z_2 &= b_2 \\
   &\vdots \\
   \ell_{n,1}z_1 + \ell_{n,2}z_2 + \ldots + \ell_{n,n}z_n &= b_n
   \end{aligned}$$

2. Then solve $L^Tx = z$ by backward substitution:
   $$\begin{aligned}
   \ell_{n,n}x_n &= z_n \\
   \ell_{n-1,n-1}x_{n-1} + \ell_{n,n-1}x_n &= z_{n-1} \\
   &\vdots \\
   \ell_{1,1}x_1 + \ell_{2,1}x_2 + \ldots + \ell_{n,1}x_n &= z_1
   \end{aligned}$$

\section{Approximate Inference}

\notes{To compute the marginal likelihood of the Gaussian process requires inverting a matrix that is the size of the data. As we have already seen this can be numerically tricky. It is also a very expensive process of cubic complexity which severely limits the size of data-sets that we can use.}

\subsection{Variational Inference}

\notes{The key idea behind variational inference is to approximate an intractable posterior distribution $p(x|y)$ with a simpler distribution $q(x)$. We do this by minimizing the KL divergence between these distributions.}

\pythonblockstart
def kl_divergence(q_mean, q_cov, p_mean, p_cov):
    """
    Compute KL divergence between two multivariate Gaussians
    
    Parameters:
    q_mean, p_mean (ndarray): Mean vectors
    q_cov, p_cov (ndarray): Covariance matrices
    
    Returns:
    float: KL(q||p)
    """
    k = len(q_mean)
    
    # Compute inverse of p_cov
    L = np.linalg.cholesky(p_cov)
    p_cov_inv = linalg.solve_triangular(L.T, 
                                      linalg.solve_triangular(L, np.eye(k), 
                                                           lower=True), 
                                      lower=False)
    
    # Compute terms
    trace_term = np.trace(p_cov_inv @ q_cov)
    mu_term = (p_mean - q_mean).T @ p_cov_inv @ (p_mean - q_mean)
    logdet_term = np.log(np.linalg.det(p_cov)) - np.log(np.linalg.det(q_cov))
    
    return 0.5 * (trace_term + mu_term - k + logdet_term)
\pythonblockend

\notes{We can derive a lower bound on the log marginal likelihood using Jensen's inequality. For a convex function $f$:}

$$f(\int g\,dx) \leq \int f \circ g\,dx$$

\notes{For the log function (which is concave), the inequality is reversed:}

$$\log(\int g\,dx) \geq \int \log(g)\,dx$$

\notes{Using this, we can derive the Evidence Lower BOund (ELBO):}

$$\log p(y) \geq \mathbb{E}_{q(f)}[\log p(y|f)] - \text{KL}(q(f)||p(f))$$

\section{Sparse Approximations}

\notes{To make GPs scalable to large datasets, we introduce inducing points - a smaller set of points that summarize the GP. Let's implement this approach:}

\pythonblockstart
def sparse_gp(X, y, Z, kernel, noise_var=1.0):
    """
    Implement sparse GP using inducing points
    
    Parameters:
    X (ndarray): Input locations (N x D)
    y (ndarray): Observations (N,)
    Z (ndarray): Inducing point locations (M x D)
    kernel: Kernel function
    noise_var: Observation noise variance
    
    Returns:
    tuple: Predictive mean and variance
    """
    # Compute kernel matrices
    Kuf = kernel(Z, X)  # M x N
    Kuu = kernel(Z, Z)  # M x M
    
    # Compute Cholesky of Kuu
    L = np.linalg.cholesky(Kuu)
    
    # Compute intermediate terms
    A = linalg.solve_triangular(L, Kuf, lower=True)
    AAT = A @ A.T
    
    # Add noise variance
    Qff = Kuf.T @ linalg.solve(Kuu, Kuf)
    
    # Compute mean and variance
    mean = Kuf.T @ linalg.solve(Kuu + AAT/noise_var, 
                               linalg.solve(Kuu, Kuf @ y))
    var = kernel(X, X) - Qff + \
          Kuf.T @ linalg.solve(Kuu + AAT/noise_var, 
                              linalg.solve(Kuu, Kuf))
    
    return mean, var
\pythonblockend

\section{Software Packages}

\notes{Several excellent software packages exist for working with Gaussian processes. Here's a brief example using GPyTorch:}

\installcode{gpytorch}

\setupcode{import gpytorch
import torch}

\pythonblockstart
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp(train_x, train_y, n_iterations=100):
    """
    Train a GP model using GPyTorch
    """
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)
    
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    model.train()
    likelihood.train()
    
    for i in range(n_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    
    return model, likelihood
\pythonblockend

\section{Summary}

\notes{In this worksheet, we've covered the practical aspects of implementing Gaussian processes:

1. Numerical stability through Cholesky decomposition
2. Approximate inference using variational methods
3. Sparse approximations for scaling to larger datasets
4. Practical implementation using modern software packages

While GPs provide an elegant mathematical framework, making them work in practice requires careful attention to computational details. The methods we've discussed here form the basis of modern GP implementations.}

\thanks

\references

