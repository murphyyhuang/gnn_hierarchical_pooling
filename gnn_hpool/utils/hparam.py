# coding=utf-8

"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import numbers
from ruamel import yaml


def _cast_to_type_if_compatible(name, param_type, value):
  """Cast hparam to the provided type, if compatible.
  Args:
    name: Name of the hparam to be cast.
    param_type: The type of the hparam.
    value: The value to be cast, if compatible.
  Returns:
    The result of casting `value` to `param_type`.
  Raises:
    ValueError: If the type of `value` is not compatible with param_type.
      * If `param_type` is a string type, but `value` is not.
      * If `param_type` is a boolean, but `value` is not, or vice versa.
      * If `param_type` is an integer type, but `value` is not.
      * If `param_type` is a float type, but `value` is not a numeric type.
  """
  fail_msg = (
      "Could not cast hparam '%s' of type '%s' from value %r" %
      (name, param_type, value))

  # Some callers use None, for which we can't do any casting/checking. :(
  if issubclass(param_type, type(None)):
    return value

  # Avoid converting a non-string type to a string.
  if (issubclass(param_type, (six.string_types, six.binary_type)) and
      not isinstance(value, (six.string_types, six.binary_type))):
    raise ValueError(fail_msg)

  # Avoid converting a number or string type to a boolean or vice versa.
  if issubclass(param_type, bool) != isinstance(value, bool):
    raise ValueError(fail_msg)

  # Avoid converting float to an integer (the reverse is fine).
  if (issubclass(param_type, numbers.Integral) and
      not isinstance(value, numbers.Integral)):
    raise ValueError(fail_msg)

  # Avoid converting a non-numeric type to a numeric type.
  if (issubclass(param_type, numbers.Number) and
      not isinstance(value, numbers.Number)):
    raise ValueError(fail_msg)

  return param_type(value)


class HParams(object):
  """
  Class to hold a set of hyper-parameters as name-value paris.
  """
  _HAS_DYNAMIC_ATTRIBUTES = True  # Required for pytype checks.

  def __init__(self, **kwargs):
    self._hparam_types = {}
    for name, value in six.iteritems(kwargs):
      self.add_hparam(name, value)

  def add_hparam(self, name, value):
    """Adds {name, value} pair to hyperparameters.
    Args:
      name: Name of the hyperparameter.
      value: Value of the hyperparameter.
    Raises:
      ValueError: if one of the arguments is invalid.
    """
    if getattr(self, name, None) is not None:
      raise ValueError('Hyperparameter name is reserved: %s' % name)
    if isinstance(value, (list, tuple)):
      if not value:
        raise ValueError('Multi-valued hyperparameters cannot be empty: %s' % name)
      self._hparam_types[name] = (type(value[0]), True)
    else:
      self._hparam_types[name] = (type(value), False)
    setattr(self, name, value)

  def set_hparam(self, name, value):
    """Set the value of an existing hyperparameter.
    This function verifies that the type of the value matches the type of the
    existing hyperparameter.
    Args:
      name: Name of the hyperparameter.
      value: New value of the hyperparameter.
    Raises:
      KeyError: If the hyperparameter doesn't exist.
      ValueError: If there is a type mismatch.
    """
    param_type, is_list = self._hparam_types[name]
    if isinstance(value, list):
      if not is_list:
        raise ValueError(
          'Must not pass a list for single-valued parameter: %s' % name)
      setattr(self, name, [
        _cast_to_type_if_compatible(name, param_type, v) for v in value])
    else:
      if is_list:
        raise ValueError(
          'Must pass a list for multi-valued parameter: %s.' % name)
      setattr(self, name, _cast_to_type_if_compatible(name, param_type, value))

  def del_hparam(self, name):
    """Removes the hyperparameter with key 'name'.
    Does nothing if it isn't present.
    Args:
      name: Name of the hyperparameter.
    """
    if hasattr(self, name):
      delattr(self, name)
      del self._hparam_types[name]

  def to_yaml(self, save_dir):
    def remove_callables(x):
      """Omit callable elements from input with arbitrary nesting."""
      if isinstance(x, dict):
        return {k: remove_callables(v) for k, v in six.iteritems(x)
                if not callable(v)}
      elif isinstance(x, list):
        return [remove_callables(i) for i in x if not callable(i)]
      return x
    with open(save_dir, 'w') as yml_writer:
      yaml.dump(remove_callables(self.values()), yml_writer)

  def from_yaml(self, read_dir):
    with open(read_dir, 'r') as yml_reader:
      hparams_dict = yaml.load(yml_reader, Loader=yaml.SafeLoader)

    for name, value in hparams_dict.items():
      self.add_hparam(name, value)

  def values(self):
    """Return the hyperparameter values as a Python dictionary.
    Returns:
      A dictionary with hyperparameter names as keys.  The values are the
      hyperparameter values.
    """
    return {n: getattr(self, n) for n in self._hparam_types.keys()}

  def __str__(self):
    return str(sorted(self.values().items()))

  def __contains__(self, key):
    return key in self._hparam_types

  def __repr__(self):
    return '%s(%s)' % (type(self).__name__, self.__str__())
