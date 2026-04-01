# SMM4H-HeaRD @ ACL-2026 Shared Task 2: Detection of Insomnia in Clinical Notes

We invite researchers and practitioners to participate in the **Shared Task 2: Detection of Insomnia in Clinical Notes** at the [**SMM4H-HeaRD @ ACL 2026**](https://healthlanguageprocessing.org/smm4h-2026/). This task focuses on developing automatic systems to identify patients potentially suffering from insomnia using electronic health records (EHRs). Designed as a text classification challenge, the task requires participants to analyze clinical notes and determine whether a patient is likely to have insomnia, as well as provide evidence supporting their predictions.

## Task Details

We have developed a comprehensive set of rules (*Insomnia rules*) to guide the identification of patients potentially suffering from insomnia. These rules incorporate both direct and indirect symptoms of insomnia, along with information about commonly prescribed medications. 

For this task, we curated an annotated corpus of clinical notes from the MIMIC-III database. Each note is annotated with:
- A binary label indicating the patient’s overall insomnia status ("yes" or "no"),  
- Rule-level labels based on the Insomnia rules, and  
- **Character-level evidence spans** from the clinical text supporting each rule-level decision.

This setup enables the development and evaluation of models that are not only accurate but also interpretable.

## Registration and Resources

To join this shared task, please register for the **Task 2 - Detection of Insomnia in Clinical Notes** through the [SMM4H-HeaRD 2026 Shared Task Registration Form](https://docs.google.com/forms/d/e/1FAIpQLSdxTPgHJOPYMeIxL94DhtfYsN1jO2Raz3NNF-b9-mIREXTgIA/viewform). Upon registration, participants will gain access to the full training, validation, and test datasets.

In addition to the data, we provide the following resources:
- The Insomnia rules: [`resources/Insomnia_Rules.md`](resources/Insomnia_Rules.md)  
- The full annotation guidelines used to annotate the corpus: [`resources/Annotation Guidelines.pdf`](resources/Annotation%20Guidelines.pdf)

These resources are essential for understanding the annotation schema and expected system outputs.

## Subtasks Description

This shared task is divided into two subtasks:

- **Subtask 1: Binary Text Classification**  
Participants are given a clinical note and must determine whether the patient described in the note is likely to suffer from insomnia ("yes" or "no").

- **Subtask 2: Multi-label Classification + Evidence Extraction**  
Participants must evaluate each clinical note according to the following Insomnia rule components: Definition 1, Definition 2, Rule B, and Rule C.  
For each component, participants must:
  - Predict a label ("yes" or "no"), and  
  - Provide supporting evidence as **character-level spans** (start–end offsets) extracted from the clinical note when the label is "yes".

*Note: For Subtask 2, Rule A is not required and will not be evaluated, as it is a deterministic combination of Definition 1 (difficulty sleeping) and Definition 2 (daytime impairment).*

## Evaluation

- **Subtask 1: Binary Text Classification**  
Performance is evaluated using the F1 score, with "yes" treated as the positive class.

- **Subtask 2: Multi-label Classification + Evidence Extraction**  
Evaluation is performed along two dimensions:
  - **Label classification**: micro-average Precision, Recall, and F1 score across all rule components.  
  - **Span extraction**: comparison of predicted and gold-standard spans using:
    - Exact Match (exact span match), and  
    - Partial Match (overlapping spans).  

For both span metrics, micro-average Precision, Recall, and F1 scores are reported across all components.

## Annotations

For each subtask, ground truth annotations are provided in JSON format. Participants must submit their system outputs following the same format as the reference annotations.

- Sample files are available at:
  - [`data/training/subtask_1`](data/training/subtask_1)  
  - [`data/training/subtask_2`](data/training/subtask_2)

For Subtask 2, each rule component must include:
- a `"label"` field ("yes"/"no"), and  
- a `"span"` field containing a list of character offsets.  

If a component is labeled "yes", the span list must be non-empty. If labeled "no", the span list must be empty (`[]`).

## Corpus

This shared task utilizes a corpus of clinical notes derived from the MIMIC-III Database. The clinical notes have been augmented with additional structured patient information, specifically sex, age, and the medications prescribed during their hospital stay.

Participants are required to complete necessary training and sign a data usage agreement to access the [MIMIC-III Clinical Database (v1.4)](https://physionet.org/content/mimiciii/1.4/). After gaining access and downloading the files, participants must run the [`text_mimic_notes.py`](text_mimic_notes.py) script to retrieve clinical notes and associated patient information using the provided note IDs. This process builds the corpus utilized in this shared task, as detailed in the instructions provided below.

### MIMIC-III Notes Processing

The `text_mimic_notes.py` Python script is designed to retrieve clinical notes and patient information from the MIMIC-III clinical database. The script takes a text file containing note IDs, and merges it with the content of the notes from MIMIC-III, including additional demographic and prescription information.

#### Requirements

- Python 3.6 or higher
- pandas library
- datetime module

#### Usage

The script requires three command-line arguments:
- `--note_ids_path`: The file path to the text file containing the note IDs.
- `--mimic_path`: The directory path containing the MIMIC-III v1.4 CSV files (`NOTEEVENTS.csv.gz`, `PRESCRIPTIONS.csv.gz` and `PATIENTS.csv.gz`).
- `--output_path`: The file path where the processed corpus CSV will be saved. This output CSV file will have two columns: the note IDs and the textual data retrieved from MIMIC-III.

#### Command Syntax

The script is executed from the command line with the following syntax:

```bash
python text_mimic_notes.py --note_ids_path [path_to_note_ids_txt] --mimic_path [path_to_mimic_csv_directory] --output_path [path_to_output_csv]
```

#### Example Command

Here is an example command that illustrates how to run the script using specific paths for each required input:

```bash
python text_mimic_notes.py --note_ids_path ./training/sample_note_ids.txt  --mimic_path ./mimic-iii/1.4 --output_path ./training/sample_corpus.csv
```

This command will process the note IDs from `./training/sample_note_ids.txt`, merge them with the data found in `./mimic-iii/1.4`, and output the resulting corpus to `./training/sample_corpus.csv`.
