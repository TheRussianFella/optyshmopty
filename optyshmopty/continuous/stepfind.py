import numpy as np

##### Step finders
# Finding the right alpha
##

class StepFinderClass:
  def __init__(self):
    pass
  def setup(self, f, gradf):
    pass
  def find(self, x: np.ndarray, h: np.ndarray):
    pass

##############################

class ConstantSF(StepFinderClass):

  def __init__(self, constant):
    self.c = constant

  def find(self, x: np.ndarray, h: np.ndarray):
    return self.c

##############################

class BacktrackingSF(StepFinderClass):

  def __init__(self, rho, alpha0, beta1, beta2):
    self.rho = rho; self.alpha0 = alpha0; self.beta1 = beta1; self.beta2 = beta2

  def setup(self, f, gradf):
    self.f = f; self.gradf = gradf

  def find(self, x: np.ndarray, h: np.ndarray):

    alpha = self.alpha0

    while (self.f(x - alpha * h) >= self.f(x) + self.beta1 * alpha * self.gradf(x).dot(h)) and \
     (self.gradf(x - alpha * h).dot(h) <= self.beta2 * self.gradf(x).dot(h)):
        alpha *= self.rho

    return alpha
