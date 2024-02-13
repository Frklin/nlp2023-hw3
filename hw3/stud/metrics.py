

def compute_metrics(preds: list, labels: list) -> tuple:
    """
    preds: predictions list [((s_start, s_end), pred_idx, (o_start, o_end))]
    labels: labels of the form [((s_start, s_end), pred_idx, (o_start, o_end))]
    indices: indices of the batch

    return: accuracy, precision, recall, f1_score
    """

    # Initialize the metrics
    accuracy = 0
    precision = 0
    recall = 0
    f1_score = 0

    correct_preds = 0
    total_preds = 0
    total_golds = 0

    # Count the correct predictions and the total predictions
    for i in range(len(preds)):
        if len(preds[i]) > 0: 
            correct_preds += len(set(preds[i]).intersection(set(labels[i])))
        total_preds += len(preds[i])
        total_golds += len(labels[i])

    # Compute the metrics
    accuracy = correct_preds / total_golds if total_golds > 0 else 0.0
    precision = correct_preds / total_preds if total_preds > 0 else 0.0
    recall = correct_preds / total_golds if total_golds > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0
<<<<<<< HEAD
    
=======

>>>>>>> dev
    return accuracy, precision, recall, f1_score

