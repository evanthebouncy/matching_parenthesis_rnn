import random
import numpy as np

def single_pos_data_gen(m):
  def _data_gen(n):
    if n <= 2:
      return [1,2]
    else:
      coin = random.random()
      branch_mass = float(n) / (2 * float(m))
      if 0.0 < coin < branch_mass:
        rec1 = _data_gen(n / 2 - 1)
        rec2 = _data_gen(n / 2 - 1)
        return rec1 + rec2
      if branch_mass < coin < branch_mass * 2.0:
        rec1 = _data_gen(n - 2)
        return [1] + rec1 + [2]
      else:
        return [1,2]
  return _data_gen(m)

def check(lst):
  balance = 0
  for x in lst:
    if balance < 0:
      return False
    else:
      if x == 1:
        balance += 1
      else:
        balance -= 1
  return balance == 0

def single_neg_data_gen(m):
  rand_ary = [random.randint(1,2) for i in range(random.randint(1, m))]
  if not check(rand_ary):
    return rand_ary
  else:
    return single_neg_data_gen(m)

def pad_data(data, data_l):
  return data + [0 for i in range(0, data_l - len(data))]

def data_to_tup(raw_data):
  def pt_xform(x):
    if x == 0:
      return [1., 0., 0.]
    if x == 1:
      return [0., 1., 0.]
    if x == 2:
      return [0., 0., 1.]
  return [pt_xform(x) for x in raw_data]

def gen_data_batch(batchsize, examplesize, pos_neg = None):
  dataz = []
  labelz = []
  for i in range(0, batchsize):
    label_i = random.random() > 0.5
    if pos_neg == True:
      label_i = True
    if pos_neg == False:
      label_i = False
    data_i = single_pos_data_gen(examplesize) if label_i else single_neg_data_gen(examplesize)
    dataz.append(data_to_tup(pad_data(data_i, examplesize)))
    labelz.append([1., 0.] if label_i else [0., 1.])
  return np.array(dataz, np.float32), np.array(labelz, np.float32)

    
