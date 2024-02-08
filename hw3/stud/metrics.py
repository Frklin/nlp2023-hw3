

def compute_metrics(preds, labels):
    """
    preds: [preds]
    labels: [((s_start, s_end), pred_idx, (o_start, o_end))]
    indices: indices of the batch

    return: accuracy, precision, recall, f1_score
    """
    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0

    correct_preds = 0
    total_preds = 0
    total_golds = 0

    for i in range(len(preds)):
        if len(preds[i]) > 0: 
            correct_preds += len(set(preds[i]).intersection(set(labels[i])))
        total_preds += len(preds[i])
        total_golds += len(labels[i])


    accuracy = correct_preds / total_golds if total_golds > 0 else 0.0
    precision = correct_preds / total_preds if total_preds > 0 else 0.0
    recall = correct_preds / total_golds if total_golds > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
    
    return accuracy, precision, recall, f1_score

