import pytest
import numpy as np
from .dirfind import *
from .stepfind import *
from .optimizer import *

@pytest.fixture
def quadratic_100():
    '''
    Get a quadratic test case.
    '''
    N = 100
    np.random.seed(42)

    # Generating a positivelydefined, symmetric matrix
    U = np.random.rand(N, N)
    Q, _ = np.linalg.qr(U)
    A = Q.dot(np.diagflat(np.random.uniform(1, 100, N))).dot(Q.T)
    A = (A + A.T) * 0.5

    b = np.random.randn(N)

    f = lambda x: x.T @ A @ x + b.T @ x
    gradf = lambda x: A @ x + b
    hesf = lambda x: A

    return {'f': f, 'gradf': gradf, 'hessf': hesf}

def test_all_simple(quadratic_100):

    tol = 0.15
    max_iter = 200

    optimizers = {
    "Gradient Descent": IterativeGradientOptimizer(GradientDF(), ConstantSF(9e-3)),
    "Adaptive GD": IterativeGradientOptimizer(GradientDF(), BacktrackingSF(alpha0=1e-1, beta1=0.1, beta2=0.4, rho=0.5)),
    "Conjugate Gradient": IterativeGradientOptimizer(ConjugateGradientDF(), ConstantSF(1e-2)),
    "CG, restarts": IterativeGradientOptimizer(ConjugateGradientDF(restart=15), ConstantSF(9e-3)),
    "Heavy ball": IterativeGradientOptimizer(HeavyBallDF(ConstantSF(1e-2), ConstantSF(0.9)), ConstantSF(1)),
    "Nesterov": IterativeGradientOptimizer(NesterovDF(BacktrackingSF(alpha0=1e-2, beta1=0.1, beta2=0.4, rho=0.5)), ConstantSF(1)),
    "Newton": IterativeGradientOptimizer(SecondOrderNewtonDF(), ConstantSF(1)),
    "BFSGD": IterativeGradientOptimizer(BFGSDF(H0=np.identity(100)), ConstantSF(2e-2))
    }

    np.random.seed(42)
    x0 = np.random.randn(100)

    for name, opt in optimizers.items():
        task = quadratic_100.copy()
        if name is not "Newton":
            del task['hessf']
        _, hist = opt.optimize(task, x0, max_iter=max_iter, tol=tol)
        assert hist[-1] < tol, "{} failed on a simple task".format(name)
