from utils import LabelFormat, Entity, reconstruct_relations_from_matrices
import config



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


# def compute_metrics(input_ids_batch, h_preds_batch, t_preds_batch, span_preds_batch, labels):
#     """
#     h_pred: head prediction matrix (batch_size, max_len, max_len)
#     t_pred: tail prediction matrix (batch_size, max_len, max_len)
#     span_pred: span prediction matrix (batch_size, max_len, max_len)
#     labels: h_true, t_true, span_true (batch_size, max_len, max_len)

#     return: accuracy, precision, recall, f1_score
#     """
#     accuracy = 0
#     precision = 0
#     recall = 0
#     f1_score = 0

#     preds: [LabelFormat] = []
    
#     # convert matrix in Paried-Entity-Relation format
#     for i in range(config.BATCH_SIZE):
#         h_triplets = find_triplets(input_ids_batch[i], h_preds_batch[i])
#         t_triplets = find_triplets(input_ids_batch[i], t_preds_batch[i])
#         span_triplets = find_triplets(input_ids_batch[i], span_preds_batch[i])

#         # Join the h_triplets with t_triplets
#         preds.append(join_triplets(h_triplets, t_triplets, span_triplets))        


#     correct_preds = 0
#     total_preds = 0
#     total_golds = 0

#     for i in range(config.BATCH_SIZE):
#         correct_preds += len(set(preds[i]).intersection(set(labels["spo"][i])))
#         total_preds += len(preds[i])
#         total_golds += len(labels["spo"][i])

#     accuracy = correct_preds / total_golds if total_golds > 0 else 0.0
#     precision = correct_preds / total_preds if total_preds > 0 else 0.0
#     recall = correct_preds / total_golds if total_golds > 0 else 0.0
#     f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0.0


#     return accuracy, precision, recall, f1_score


def find_triplets(input_ids, matrix):
    e2e = set()
    e2r = set()
    r2e = set()

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0 or i == j:
                continue
            if i < config.MAX_LEN and j < config.MAX_LEN:
                e2e.add(((i, input_ids[i].item()),(j, input_ids[j].item())))
            elif i < config.MAX_LEN:
                e2r.add(((i, input_ids[i].item()), j-config.MAX_LEN))
            else:
                r2e.add((i-config.MAX_LEN, (j, input_ids[j].item())))

    triplets = construct_triplets(e2e, e2r, r2e)

    return triplets


def construct_triplets(e2e, e2r, r2e):
    triplets = []

    for subject, relation in e2r:
        objects = [tup[1] for tup in r2e if tup[0] == relation]
        for object in objects:
            if (subject, object) in e2e:
                triplets.append((subject[0], relation, object[0]))

    return triplets


def join_triplets(h_triplets, t_triplets, span_triplets):
    triplets = []
    for h_subj, h_rel, h_obj in h_triplets:
        filtered_t_triplets = [tup for tup in t_triplets if tup[1] == h_rel and tup[0] > h_subj and tup[2] > h_obj]
        if len(filtered_t_triplets) == 0:
            continue

        for t_subj, _, t_obj in sorted(list(filtered_t_triplets), key=lambda x: x[0]):
            triplets.append(LabelFormat(
                                subject=Entity(
                                    start_idx=h_subj,
                                    end_idx=t_subj,
                                    entity_type="subject"
                                ),
                                relation=h_rel,
                                object=Entity(
                                    start_idx=h_obj,
                                    end_idx=t_obj,
                                    entity_type="object"
                                )))
            
    return triplets
            