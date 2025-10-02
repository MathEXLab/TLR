import os
import sys
import types


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class _FakeArray:
    def __init__(self, data):
        self.data = self._to_nested_lists(data)

    @staticmethod
    def _to_nested_lists(value):
        if isinstance(value, _FakeArray):
            return _FakeArray._to_nested_lists(value.data)
        if isinstance(value, (list, tuple)):
            return [ _FakeArray._to_nested_lists(item) for item in value ]
        return value

    @property
    def ndim(self):
        if isinstance(self.data, list) and self.data and isinstance(self.data[0], list):
            return 2
        if isinstance(self.data, list):
            return 1
        return 0

    @property
    def shape(self):
        if self.ndim == 2:
            rows = len(self.data)
            cols = len(self.data[0]) if rows else 0
            return (rows, cols)
        if self.ndim == 1:
            return (len(self.data),)
        return ()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        if shape == (-1, 1):
            if self.ndim != 1:
                raise NotImplementedError
            return _FakeArray([[item] for item in self.data])
        raise NotImplementedError

    def _binary_op(self, other, op):
        other_data = other.data if isinstance(other, _FakeArray) else other
        if self.ndim == 2:
            if isinstance(other_data, list) and other_data and isinstance(other_data[0], list):
                # Support broadcasting (n, m) vs (n, 1).
                if len(other_data[0]) == 1 and len(other_data) == len(self.data):
                    result = [
                        [op(val, other_row[0]) for val in row]
                        for row, other_row in zip(self.data, other_data)
                    ]
                else:
                    result = [
                        [op(val, other_row[j]) for j, val in enumerate(row)]
                        for row, other_row in zip(self.data, other_data)
                    ]
            else:
                result = [
                    [op(val, other_data) for val in row]
                    for row in self.data
                ]
            return _FakeArray(result)
        if self.ndim == 1:
            if isinstance(other_data, list):
                result = [op(a, b) for a, b in zip(self.data, other_data)]
            else:
                result = [op(a, other_data) for a in self.data]
            return _FakeArray(result)
        return op(self.data, other_data)

    def __gt__(self, other):
        return self._binary_op(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._binary_op(other, lambda a, b: a >= b)

    def __eq__(self, other):
        return self._binary_op(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._binary_op(other, lambda a, b: a != b)

    def __iter__(self):
        if self.ndim == 0:
            yield self.data
        else:
            for item in self.data:
                if isinstance(item, list):
                    yield _FakeArray(item)
                else:
                    yield item

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
            rows = self.data[row_key]
            if isinstance(row_key, slice):
                raise NotImplementedError
            if isinstance(col_key, slice):
                return _FakeArray(rows[col_key])
            return rows[col_key]
        result = self.data[key]
        if isinstance(result, list):
            return _FakeArray(result)
        return result

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row_key, col_key = key
            if isinstance(row_key, slice):
                raise NotImplementedError
            if isinstance(col_key, slice):
                raise NotImplementedError
            self.data[row_key][col_key] = value
        else:
            if isinstance(self.data[key], list):
                raise NotImplementedError
            self.data[key] = value


def _fake_array(data, dtype=None):
    return _FakeArray(data)


def _fake_sum(array, axis=None):
    data = array.data if isinstance(array, _FakeArray) else array
    if axis is None:
        if isinstance(data, list) and data and isinstance(data[0], list):
            return sum(sum(1 if val else 0 for val in row) for row in data)
        if isinstance(data, list):
            return sum(1 if val else 0 for val in data)
        return data
    if axis == 1:
        return _FakeArray([sum(1 if val else 0 for val in row) for row in data])
    raise NotImplementedError


def _fake_where(condition):
    data = condition.data if isinstance(condition, _FakeArray) else condition
    if isinstance(data, list) and data and isinstance(data[0], list):
        rows, cols = [], []
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                if val:
                    rows.append(i)
                    cols.append(j)
        return (rows, cols)
    if isinstance(data, list):
        return ([i for i, val in enumerate(data) if val],)
    return ([0] if data else [],)


def _fake_all(array):
    data = array.data if isinstance(array, _FakeArray) else array
    if isinstance(data, list):
        return all(bool(val) for val in data)
    return bool(data)


def _register_stub(name):
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


# Register stubs for unavailable dependencies before importing the target module.
_fake_numpy = types.ModuleType("numpy")
_fake_numpy.array = _fake_array
_fake_numpy.sum = _fake_sum
_fake_numpy.where = _fake_where
_fake_numpy.all = _fake_all
_fake_numpy.nan = float("nan")
_fake_numpy.zeros = lambda *args, **kwargs: (_FakeArray([[0]]))  # minimal placeholder
_fake_numpy.quantile = lambda *args, **kwargs: (_FakeArray([0]))
_fake_numpy.argwhere = lambda *args, **kwargs: (_FakeArray([[0, 0]]))
_fake_numpy.arange = lambda n: _FakeArray(list(range(n)))
sys.modules['numpy'] = _fake_numpy

_register_stub('xarray')
_tqdm_module = _register_stub('tqdm')
_tqdm_module.tqdm = lambda *args, **kwargs: args[0]

_scipy = _register_stub('scipy')
_scipy_stats = types.ModuleType('scipy.stats')
_scipy.stats = _scipy_stats
sys.modules['scipy.stats'] = _scipy_stats

_statsmodels = _register_stub('statsmodels')
_statsmodels_api = types.ModuleType('statsmodels.api')
_statsmodels.api = _statsmodels_api
sys.modules['statsmodels.api'] = _statsmodels_api

_sklearn = _register_stub('sklearn')
_sklearn_metrics = types.ModuleType('sklearn.metrics')
_sklearn.metrics = _sklearn_metrics
sys.modules['sklearn.metrics'] = _sklearn_metrics
_sklearn_pairwise = types.ModuleType('sklearn.metrics.pairwise')
_sklearn_metrics.pairwise = _sklearn_pairwise
sys.modules['sklearn.metrics.pairwise'] = _sklearn_pairwise


from scripts.local_indices import _correct_n_neigh


def test_correct_n_neigh_adds_edge_neighbors():
    dist_log = _fake_array([
        [0.0, 0.5, 0.2],
        [0.0, 0.5, 0.5],
    ])
    q = _fake_array([0.2, 0.5])
    n_neigh = 1

    exceeds_bool = dist_log > q.reshape(-1, 1)
    assert _fake_sum(exceeds_bool[-1]) == 0

    corrected = _correct_n_neigh(exceeds_bool, dist_log, q, n_neigh)

    neighbour_counts = _fake_sum(corrected, axis=1)
    equals_target = neighbour_counts == n_neigh
    assert _fake_all(equals_target)
