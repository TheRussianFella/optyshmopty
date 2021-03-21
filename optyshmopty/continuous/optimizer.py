import numpy as np
import inspect

from .dirfind import DirectionFinderClass
from .stepfind import StepFinderClass

##### Main routine class

class IterativeGradientOptimizer:

  def __init__(self, direction_finder: DirectionFinderClass, step_finder: StepFinderClass):
    self.direction_finder = direction_finder
    self.step_finder = step_finder

    # Check if direction finder needs to update after a step
    self.update = callable(getattr(direction_finder, "update", None))

    # Delete unnecassary function info
    self.desired_info = inspect.getfullargspec(direction_finder.setup).args

  def optimize(self, func_info, x0, max_iter=100, tol=1e-8, save_history=True):
    '''
    func_info: dictionary with information about a function, that a chosen method
    will need - has to include f and gradf. (check that naming of the arguments match)
    '''

    func_info = {key: func_info[key] for key in filter(lambda x: x in self.desired_info, func_info.keys())}
    gradf = func_info['gradf']

    self.direction_finder.setup(**func_info, x0=x0)
    self.step_finder.setup(f=func_info['f'], gradf=func_info['gradf'])

    history = []
    x = x0
    it = 0; gradn = np.linalg.norm(gradf(x))

    while gradn > tol and it < max_iter:

      h = self.direction_finder.find(x)
      alpha = self.step_finder.find(x, h)

      if self.update:
          self.direction_finder.update(x, alpha, h)

      x = x - alpha * h

      if save_history:
        history.append(gradn)
      gradn = np.linalg.norm(gradf(x)); it += 1

    history.append(gradn)

    return (x, history)
