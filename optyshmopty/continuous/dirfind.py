from .stepfind import StepFinderClass
import numpy as np

#### Direction finders
# Theese are classes that find direction in
# x_{k+1} = x_k + \alpha * h
###

class DirectionFinderClass:
  def __init__(self):
    pass

  def setup(self, f, gradf, x0):
    self.f = f; self.gradf = gradf

  def find(self, x):
    pass

##############################

class GradientDF(DirectionFinderClass):

  def find(self, x):
    return self.gradf(x)

##############################

class ConjugateGradientDF(DirectionFinderClass):

  def __init__(self, restart:int = None):
    self.restart = restart
    self.counter = 0

  def setup(self, f, gradf, x0):
    self.f = f; self.gradf = gradf
    self.grad = gradf(x0)
    self.p = -self.grad
    self.first_step = True # TODO: come up with something more intellegent

  def find(self, x):

    if self.first_step:
      self.first_step = False

    else:
      grad_next = self.gradf(x)
      beta = grad_next.dot(grad_next) / self.grad.dot(self.grad)
      self.p = -grad_next + beta * self.p
      self.grad = grad_next.copy()

      if self.restart and self.counter % self.restart == 0:
        self.grad = self.gradf(x)
        self.p = -self.grad

    self.counter += 1
    return -self.p # so that update rule is consistent across methods

##############################

class HeavyBallDF(DirectionFinderClass):

  def __init__(self, alpha_finder: StepFinderClass, beta_finder: StepFinderClass):
    self.alpha_finder = alpha_finder; self.beta_finder = beta_finder

  def setup(self, f, gradf, x0):
    self.prev_x = x0; self.f = f; self.gradf = gradf
    self.alpha_finder.setup(f, gradf)
    self.beta_finder.setup(f, gradf)

  def find(self, x):
    grad = self.gradf(x)
    beta = self.beta_finder.find(x, grad)

    alpha = self.alpha_finder.find(x + beta * (x - self.prev_x), grad);

    h = alpha * grad - beta * (x - self.prev_x)
    self.prev_x = x

    return h

##############################

class NesterovDF(DirectionFinderClass):

  def __init__(self, alpha_finder: StepFinderClass):
    self.alpha_finder = alpha_finder

  def setup(self, f, gradf, x0):
    self.f = f; self.gradf = gradf
    self.prev_x = x0
    self.alpha_finder.setup(f, gradf)
    self.counter = 1

  def find(self, x):

    # Choosing a step (choosing it at a point, from which we are making a gradient step)

    yk = x + (self.counter - 1)/(self.counter + 2) * (x - self.prev_x)
    step_dir = self.gradf(yk)

    step = self.alpha_finder.find(yk, step_dir)

    # Building a direction
    h = (yk - x) - step * step_dir
    self.prev_x = x
    self.counter += 1

    return -h


#### Second order methods #####

class SecondOrderNewtonDF(DirectionFinderClass):

  def __init__(self, hsolver=None):

    if hsolver is None:
      self.hsolver = lambda x: np.linalg.solve(self.hessf(x), -self.gradf(x))
    else:
      self.hsolver = hsolver

  def setup(self, f, gradf, hessf, x0):
    self.f = f; self.gradf = gradf; self.hessf = hessf; self.x0 = x0

  def find(self, x):

    h = self.hsolver(x)
    return -h

class BFGSDF(DirectionFinderClass):

    def __init__(self, H0):
        self.H = H0

    def setup(self, f, gradf, x0):
        self.f = f; self.gradf = gradf; self.x0 = x0

    def find(self, x):
        return self.H.dot(self.gradf(x))

    def update(self, x, alpha, h):

        x_next = x - alpha * h

        y = self.gradf(x_next) - self.gradf(x)
        s = -alpha * h
        rho = 1 / y.dot(s)

        # Multiply everything in a right way, so that complexity is O(n^2)
        Hy = self.H.dot(y)
        Hys = np.outer(Hy, s)
        ss = np.outer(s, s)

        self.H = rho * ss + self.H - rho * Hys - rho * Hys.T + rho**2 * y.dot(Hy) * ss
