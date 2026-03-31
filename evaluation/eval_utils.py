import json
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


# Define class mappings
POSITIVE_CLASS = 'yes'
NEGATIVE_CLASS = 'no'
MAP_CLASS = {
    POSITIVE_CLASS: 1,
    NEGATIVE_CLASS: 0
}


def check_gs_pred_note_ids(df_gs, df_pred):
    """
    Check if all note IDs in the gold standard are present in the predictions.

    Args:
        df_gs (DataFrame): DataFrame containing the Gold Standard annotations
        df_pred (DataFrame): DataFrame containing the system predictions

    Raises:
        ValueError: If note IDs in the gold standard are missing from the predictions
    """
    missing_ids = df_gs.index.difference(df_pred.index)
    if not missing_ids.empty:
        raise ValueError(
            f"Missing the following note IDs in predictions: {', '.join(map(str, missing_ids))}"
        )


def check_dict_note_ids(dict_gs, dict_pred):
    """
    Check if all note IDs in the gold standard dictionary are present in the predictions dictionary.

    Args:
        dict_gs (dict): Dictionary containing the Gold Standard annotations
        dict_pred (dict): Dictionary containing the system predictions

    Raises:
        ValueError: If note IDs in the gold standard are missing from the predictions
    """
    missing_ids = set(dict_gs.keys()) - set(dict_pred.keys())
    if missing_ids:
        raise ValueError(f"Missing note IDs in predictions: {', '.join(map(str, missing_ids))}")


def read_and_validate_json(json_path, labels, data_name='system predictions'):
    """
    Generic function to read and validate JSON data.

    Args:
        json_path (str): Path to the JSON file
        labels (list): List of labels to validate in the JSON data
        data_name (str): Description of the data (either 'gold standard' or 'system predictions')

    Returns:
        DataFrame: DataFrame with note IDs as indices and required labels as columns

    Raises:
        ValueError: If the JSON data is malformed or labels are incorrect
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    validated_data = {}
    for note_id, content in data.items():
        validated_data[note_id] = {}
        for label in labels:
            if label not in content:
                raise ValueError(f"Missing '{label}' label for note {note_id} in {data_name}")
            
            label_value = content[label].lower()
            if label_value not in [POSITIVE_CLASS, NEGATIVE_CLASS]:
                raise ValueError(
                    f"Incorrect value for '{label}' label at note {note_id} in {data_name}: {content[label]}. "
                    f"Expected '{POSITIVE_CLASS}' or '{NEGATIVE_CLASS}'."
                )
            validated_data[note_id][label] = MAP_CLASS[label_value]

    return pd.DataFrame.from_dict(validated_data, orient='index')


def subtask_2_read_and_validate_json(json_path, labels, data_name='system predictions'):
    """
    Function to read and validate JSON data from Subtask 2, ensuring each item's format is correct,
    and that the character spans align with the assigned labels.

    Args:
        json_path (str): Path to the JSON file
        labels (list): List of labels to validate in the JSON data
        data_name (str): Description of the data (either 'gold standard' or 'system predictions')

    Returns:
        dict: Dictionary with note IDs as keys and validated label and span data as values

    Raises:
        ValueError: If the JSON data is malformed or labels are incorrect
    """
    with open(json_path, 'r') as file:
        data = json.load(file)

    validated_data = {}
    for note_id, content in data.items():
        validated_data[note_id] = {}
        for label in labels:
            if label not in content:
                raise ValueError(f"Missing '{label}' label for note {note_id} in {data_name}")

            label_content = content[label]
            if not isinstance(label_content, dict):
                raise ValueError(
                    f"Incorrect format for '{label}' label at note {note_id} in {data_name}: {label_content}. "
                    f"Expected dict with 'label' and 'span' keys."
                )

            # 'label' key
            if 'label' not in label_content:
                raise ValueError(f"Missing 'label' key in '{label}' for note {note_id} in {data_name}")
            label_content_value = label_content['label'].lower()
            if label_content_value not in [POSITIVE_CLASS, NEGATIVE_CLASS]:
                raise ValueError(
                    f"Incorrect 'label' for '{label}' at note {note_id} in {data_name}: {label_content['label']}. "
                    f"Expected '{POSITIVE_CLASS}' or '{NEGATIVE_CLASS}'."
                )

            # 'span' key
            if 'span' not in label_content:
                raise ValueError(f"Missing 'span' key in '{label}' for note {note_id} in {data_name}")
            if not isinstance(label_content['span'], list) or not all(isinstance(i, str) for i in label_content['span']):
                raise ValueError(
                    f"Incorrect 'span' format for '{label}' at note {note_id} in {data_name}: {label_content['span']}. "
                    f"Expected list of strings."
                )

            # Check span list length
            if len(label_content['span']) > 20:
                raise ValueError(
                    f"Too many spans for '{label}' at note {note_id} in {data_name}: {len(label_content['span'])} spans found. "
                    f"Maximum allowed is 20."
                )

            # Check 'label' and 'span' consistency
            if label_content_value == POSITIVE_CLASS and len(label_content['span']) == 0:
                raise ValueError(
                    f"Empty 'span' list for positive '{label}' at note {note_id} in {data_name}. "
                    f"Expected non-empty list."
                )
            if label_content_value == NEGATIVE_CLASS and len(label_content['span']) > 0:
                raise ValueError(
                    f"Non-empty 'span' list for negative '{label}' at note {note_id} in {data_name}. "
                    f"Expected empty list."
                )

            # Validate span format - each span should be parseable as "start end" or "start end;start end"
            for span_str in label_content['span']:
                try:
                    for part in span_str.split(";"):
                        parts = part.strip().split()
                        if len(parts) != 2:
                            raise ValueError("Expected exactly two integers")
                        int(parts[0])  # start
                        int(parts[1])  # end
                except (ValueError, IndexError):
                    raise ValueError(
                        f"Incorrect 'span' format for '{label}' at note {note_id} in {data_name}: {label_content['span']}. "
                        f"Expected format: 'start end' or 'start end;start end' with positiveinteger values."
                    )

            # Store the validated label and span
            validated_data[note_id][label] = {
                'label': label_content_value,
                'span': label_content['span'],
            }

    return validated_data


# -----------------------------
# Classification Metrics
# -----------------------------

def compute_metrics(labels, preds, average='binary', target_names=[NEGATIVE_CLASS, POSITIVE_CLASS]):
    """
    Compute precision, recall, and F1-score for the given labels and predictions.

    Args:
        labels (Series): True labels
        preds (Series): Predicted labels
        average (str): Type of averaging performed on the metrics depending on the task.
            Expected values are: 'binary' (Subtask 1) or 'micro' (Subtask 2A)
        target_names (list): List of target names for classification report

    Returns:
        dict: Dictionary containing precision, recall, and F1-score
    """
    print(classification_report(
        y_true=labels,
        y_pred=preds,
        digits=4,
        zero_division=0.0,
        target_names=target_names
    ))
    return {
        'Precision': round(
            precision_score(
                y_true=labels,
                y_pred=preds,
                average=average
            ), 4
        ),
        'Recall': round(
            recall_score(
                y_true=labels,
                y_pred=preds,
                average=average
            ), 4
        ),
        'F1-score': round(
            f1_score(
                y_true=labels,
                y_pred=preds,
                average=average
            ), 4
        ),
    }


# -----------------------------
# Span Utilities (for Subtask 2)
# -----------------------------

def parse_span(span_str):
    """
    Parse a single span string into a list of (start, end) tuples.

    Supports:
    - "start end" -> [(start, end)]
    - "start end;start end" -> [(start1, end1), (start2, end2)]

    Args:
        span_str (str): String representation of one or more character spans

    Returns:
        list: List of (start, end) tuples
    """
    segments = []
    for part in span_str.split(";"):
        s, e = map(int, part.strip().split())
        segments.append((s, e))
    return segments


def parse_spans(span_list):
    """
    Parse a list of span strings.

    Args:
        span_list (list): List of span strings

    Returns:
        list: List of parsed spans (each span is a list of (start, end) tuples)
    """
    return [parse_span(s) for s in span_list]


def segment_overlap(a, b):
    """
    Calculate character overlap between two segments.

    Args:
        a (tuple): (start, end) of first segment
        b (tuple): (start, end) of second segment

    Returns:
        int: Number of overlapping characters
    """
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def span_overlap(span1, span2):
    """
    Calculate total character overlap between two spans (which may contain multiple segments).

    Args:
        span1 (list): List of (start, end) tuples
        span2 (list): List of (start, end) tuples

    Returns:
        int: Total number of overlapping characters
    """
    total = 0
    for s1 in span1:
        for s2 in span2:
            total += segment_overlap(s1, s2)
    return total


def exact_match(span1, span2):
    """
    Check if two spans are exactly identical.

    Args:
        span1 (list): List of (start, end) tuples
        span2 (list): List of (start, end) tuples

    Returns:
        bool: True if spans are identical
    """
    if len(span1) != len(span2):
        return False
    return all(s1 == s2 for s1, s2 in zip(span1, span2))


def match_exact(pred_spans, gold_spans):
    """
    Match predicted spans to gold spans using exact matching.
    Each gold span can be matched at most once.

    Args:
        pred_spans (list): List of predicted spans
        gold_spans (list): List of gold standard spans

    Returns:
        tuple: (true_positives, false_positives, false_negatives)
    """
    matched = set()
    tp = 0

    for p in pred_spans:
        for i, g in enumerate(gold_spans):
            if i in matched:
                continue
            if exact_match(p, g):
                tp += 1
                matched.add(i)
                break

    fp = len(pred_spans) - tp
    fn = len(gold_spans) - tp
    return tp, fp, fn


def match_partial(pred_spans, gold_spans):
    """
    Match predicted spans to gold spans using partial (overlap-based) matching.
    Each predicted span is matched to the gold span with maximum overlap.
    Each gold span can be matched at most once.

    Args:
        pred_spans (list): List of predicted spans
        gold_spans (list): List of gold standard spans

    Returns:
        tuple: (true_positives, false_positives, false_negatives)
    """
    matched = set()
    tp = 0

    for p in pred_spans:
        best_idx = None
        best_overlap = 0

        for i, g in enumerate(gold_spans):
            if i in matched:
                continue

            overlap = span_overlap(p, g)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i

        if best_overlap > 0:
            tp += 1
            matched.add(best_idx)

    fp = len(pred_spans) - tp
    fn = len(gold_spans) - tp
    return tp, fp, fn


def compute_prf(tp, fp, fn):
    """
    Compute precision, recall, and F1-score from counts.

    Args:
        tp (int): True positives
        fp (int): False positives
        fn (int): False negatives

    Returns:
        tuple: (precision, recall, f1)
    """
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1
