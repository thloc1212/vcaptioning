"""One-to-one event localization metrics for exploratory boundary evaluation."""

THRESHOLDS = (0.3, 0.5, 0.7, 0.9)


def interval_iou(a, b):
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return overlap / union if union > 0 else 0.0


def counts(predictions, references, threshold):
    """Maximum cardinality matching; each prediction and GT can match once."""
    edges = [[j for j, ref in enumerate(references) if interval_iou(pred, ref) >= threshold]
             for pred in predictions]
    owner = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False

    matched = sum(augment(i, set()) for i in range(len(predictions)))
    return matched, len(predictions), len(references)


def summarize(all_predictions, all_references, thresholds=THRESHOLDS):
    result = {}
    for threshold in thresholds:
        matched = predicted = reference = 0
        for predictions, references in zip(all_predictions, all_references):
            m, p, r = counts(predictions, references, threshold)
            matched += m
            predicted += p
            reference += r
        precision = matched / predicted if predicted else 0.0
        recall = matched / reference if reference else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[str(threshold)] = {"precision": precision, "recall": recall, "f1": f1,
                                  "matched": matched, "predictions": predicted, "references": reference}
    return result
