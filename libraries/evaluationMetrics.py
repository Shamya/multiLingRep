def print_accuracy_fscore(Y_pred, Y_test):
  count = 0.0
  correct = 0.0
  scores = {'tp':0.0,'tn':0.0,'fp':0.0,'fn':0.0}

  for i in range(len(Y_pred)):
      count += 1
      if Y_pred[i] == Y_test[i]:
          correct += 1 

      #precision and recall
      #true positive 
      if Y_test[i] == 1:
          if Y_pred[i] == 1:
              scores['tp'] += 1
          else:
              scores['fn'] += 1
      else:
          if Y_pred[i] == 1:
              scores['fp'] += 1
          else:
              scores['tn'] += 1
  
  print scores
  test_acc = correct / count 
  if(scores['tp']==0):
    precision = 0
    recall = 0
    f_score = 0
  else:
    precision = scores['tp'] / (scores['tp'] + scores['fp'])
    recall = scores['tp'] / (scores['tp'] + scores['fn'])
    f_score = (2*precision*recall)/(precision+recall)
  print "Precision : ", precision
  
  print "Recall : ", recall
  
  print "F score : ", f_score
  
  print "Accuracy : ", test_acc
