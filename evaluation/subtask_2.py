import argparse
import pandas as pd
import eval_utils


def main(gs_path, pred_path):
    """
    Main function to evaluate system predictions against gold standard annotations for Subtask 2.

    Args:
        gs_path (str): Path to the JSON file containing the Gold Standard annotations
        pred_path (str): Path to the JSON file containing the system predictions
    """
    arr_labels = ['Definition 1', 'Definition 2', 'Rule B', 'Rule C']

    # Load and validate data from JSON files
    dict_gs = eval_utils.subtask_2_read_and_validate_json(
        json_path=gs_path,
        data_name='gold standard',
        labels=arr_labels
    )
    dict_pred = eval_utils.subtask_2_read_and_validate_json(
        json_path=pred_path,
        data_name='system predictions',
        labels=arr_labels
    )

    # Check for matching note IDs between gold standard and predictions
    eval_utils.check_dict_note_ids(dict_gs, dict_pred)

    # -----------------------------
    # Part 1: Label Classification Metrics
    # -----------------------------
    print("\n" + "="*80)
    print("PART 1: LABEL CLASSIFICATION METRICS")
    print("="*80)

    # Convert to binary labels for classification evaluation
    gs_labels = {}
    pred_labels = {}
    for note_id in dict_gs:
        gs_labels[note_id] = {}
        pred_labels[note_id] = {}
        for label in arr_labels:
            gs_labels[note_id][label] = eval_utils.MAP_CLASS[dict_gs[note_id][label]['label']]
            pred_labels[note_id][label] = eval_utils.MAP_CLASS[dict_pred[note_id][label]['label']]

    # Create DataFrames for classification metrics
    df_gs = pd.DataFrame.from_dict(gs_labels, orient='index')
    df_pred = pd.DataFrame.from_dict(pred_labels, orient='index')

    # Compute classification metrics
    classification_result = eval_utils.compute_metrics(
        labels=df_gs.values,
        preds=df_pred.loc[df_gs.index].values,
        average='micro',
        target_names=arr_labels
    )
    print("\nLabel Classification Results:")
    print(classification_result)

    # -----------------------------
    # Part 2: Span Extraction Metrics
    # -----------------------------
    print("\n" + "="*80)
    print("PART 2: SPAN EXTRACTION METRICS")
    print("="*80)

    # Initialize results storage
    results = {}
    global_counts = {
        "exact": {"tp": 0, "fp": 0, "fn": 0},
        "partial": {"tp": 0, "fp": 0, "fn": 0}
    }

    # Evaluate each label
    for label in arr_labels:
        tp_e = fp_e = fn_e = 0
        tp_p = fp_p = fn_p = 0

        for note_id in dict_gs:
            gs_spans = eval_utils.parse_spans(dict_gs[note_id][label]['span'])
            pred_spans = eval_utils.parse_spans(dict_pred[note_id][label]['span'])

            # Exact matching
            tp, fp, fn = eval_utils.match_exact(pred_spans, gs_spans)
            tp_e += tp
            fp_e += fp
            fn_e += fn

            # Partial matching
            tp, fp, fn = eval_utils.match_partial(pred_spans, gs_spans)
            tp_p += tp
            fp_p += fp
            fn_p += fn

        # Compute per-label metrics
        p_e, r_e, f1_e = eval_utils.compute_prf(tp_e, fp_e, fn_e)
        p_p, r_p, f1_p = eval_utils.compute_prf(tp_p, fp_p, fn_p)

        results[label] = {
            "exact": {"Precision": round(p_e, 4), "Recall": round(r_e, 4), "F1": round(f1_e, 4)},
            "partial": {"Precision": round(p_p, 4), "Recall": round(r_p, 4), "F1": round(f1_p, 4)}
        }

        # Accumulate global counts for micro-average
        global_counts["exact"]["tp"] += tp_e
        global_counts["exact"]["fp"] += fp_e
        global_counts["exact"]["fn"] += fn_e

        global_counts["partial"]["tp"] += tp_p
        global_counts["partial"]["fp"] += fp_p
        global_counts["partial"]["fn"] += fn_p

    # Compute micro-average across all labels
    overall = {}
    for mode in ["exact", "partial"]:
        tp = global_counts[mode]["tp"]
        fp = global_counts[mode]["fp"]
        fn = global_counts[mode]["fn"]

        p, r, f1 = eval_utils.compute_prf(tp, fp, fn)
        overall[mode] = {"Precision": round(p, 4), "Recall": round(r, 4), "F1": round(f1, 4)}

    # Print span extraction results
    print("\nPer-label span extraction results:")
    for label in arr_labels:
        print(f"\n{label}:")
        print(f"  Exact matching   - {results[label]['exact']}")
        print(f"  Partial matching - {results[label]['partial']}")

    print("\nOverall Span Extraction (Micro-average):")
    print(f"  Exact matching   - {overall['exact']}")
    print(f"  Partial matching - {overall['partial']}")


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Evaluate predictions for Subtask 2 of the Insomnia detection task.")
    parser.add_argument(
        "-g",
        "--gs_path",
        type=str,
        required=True,
        help="File path to the JSON file containing the Gold Standard annotations"
    )
    parser.add_argument(
        "-p",
        "--pred_path",
        type=str,
        required=True,
        help="File path to the JSON file containing the system predictions"
    )

    # Parse arguments
    args = parser.parse_args()

    # Run evaluation
    main(
        gs_path=args.gs_path,
        pred_path=args.pred_path
    )