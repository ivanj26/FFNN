def binary_accuracy_score(y_true, y_pred):
  tp = 0
  tn = 0
  fn = 0
  fp = 0
  for y_t, y_p in zip(y_true, y_pred):
    if y_t == round(y_p) :
      if y_t == 0:
        tn += 1
      else:
        tp += 1
    else:
      if y_t == 0:
        fn += 1
      else:
        fp += 1
  return (tp + tn) / (tp + tn + fp +fn)