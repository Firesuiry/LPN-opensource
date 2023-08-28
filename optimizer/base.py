import logging
import abc


class optimizer:

    def __init__(self, maxfe, fun, npart, ndim, hyperparams=None):
        self.fes = 0
        self.max_fes = maxfe
        self.fun = fun
        self.npart = npart
        self.ndim = ndim
        self.run_flag = True
        self.show = False

        self.name = 'optimizer'
        self.record = {}

        if hyperparams:
            self._build(hyperparams)

    def evaluate(self, x):
        # print(f'{self.fes / self.max_fes}')
        record_num = 0.01 * self.max_fes
        if len(x.shape) == 1:
            self.fes += 1
            if self.fes % record_num == 0:
                self.data_store()
            if self.fes >= self.max_fes:
                self.run_flag = False
            return self.fun(x.reshape(1, -1))
        else:
            old_fe = self.fes
            self.fes += x.shape[0]
            if old_fe % record_num != self.fes % record_num:
                self.data_store()
            if self.fes >= self.max_fes:
                self.run_flag = False
            return self.fun(x)

    def _build(self, hyperparams):
        """This method serves as the object building process.
        Args:
            hyperparams (dict): Contains key-value parameters to the meta-heuristics.
        """

        logging.debug('Running private method: build().')

        # We need to save the hyperparams object for faster looking up
        self.hyperparams = hyperparams

        # Checks if hyperparams are really provided
        if hyperparams:
            # If one can find any hyperparam inside its object
            for k, v in hyperparams.items():
                # Set it as the one that will be used
                setattr(self, k, v)

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        print('Algorithm: %s | Hyperparameters: %s | '
              'Built: %s.' % (
                  self.name, str(hyperparams),
                  self.built))

    def better(self, fit1, fit2):
        return fit1 > fit2

    @abc.abstractmethod
    def show_method(self):
        pass

    def data_store(self):
        self.record[self.fes] = {
            'best': self.get_best_value(),
        }

    @abc.abstractmethod
    def get_best_value(self):
        pass

    def show(self):
        pass
