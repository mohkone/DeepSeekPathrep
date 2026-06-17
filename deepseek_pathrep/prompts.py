"""Prompt builders for a DeepSeek replication of the PathRep-Bench tasks."""

PROMPT_VERSION = "stage_explicit_substage_v2"

CANCER_TYPES = [
    "Adrenocortical carcinoma",
    "Bladder urothelial carcinoma",
    "Brain lower grade glioma",
    "Breast invasive carcinoma",
    "Cervical squamous cell carcinoma and endocervical adenocarcinoma",
    "Cholangiocarcinoma",
    "Colon adenocarcinoma",
    "Esophageal carcinoma",
    "Glioblastoma multiforme",
    "Head and neck squamous cell carcinoma",
    "Kidney chromophobe",
    "Kidney renal clear cell carcinoma",
    "Kidney renal papillary cell carcinoma",
    "Liver hepatocellular carcinoma",
    "Lung adenocarcinoma",
    "Lung squamous cell carcinoma",
    "Lymphoid neoplasm diffuse large B-cell lymphoma",
    "Mesothelioma",
    "Ovarian serous cystadenocarcinoma",
    "Pancreatic adenocarcinoma",
    "Pheochromocytoma and paraganglioma",
    "Prostate adenocarcinoma",
    "Rectum adenocarcinoma",
    "Sarcoma",
    "Skin cutaneous melanoma",
    "Stomach adenocarcinoma",
    "Testicular germ cell tumors",
    "Thymoma",
    "Thyroid carcinoma",
    "Uterine carcinosarcoma",
    "Uterine corpus endometrial carcinoma",
    "Uveal melanoma",
]

AJCC_STAGE_OPTIONS = ["Stage I", "Stage II", "Stage III", "Stage IV"]


def _json_system(extra_instruction):
    return (
        "You are a pathology NLP assistant for retrospective research. "
        "Use only the pathology report content supplied by the user. "
        "Return valid json only, with no markdown, no commentary, and no extra keys. "
        "If details are ambiguous, choose the best supported option from the allowed labels. "
        f"{extra_instruction}"
    )


def cancer_type_messages(report_text, cancer_types=None):
    labels = cancer_types or CANCER_TYPES
    return [
        {
            "role": "system",
            "content": _json_system(
                'Output exactly this schema: {"cancer_type": "<one allowed label>"}'
            ),
        },
        {
            "role": "user",
            "content": (
                "Identify the cancer type in this pathology report.\n\n"
                "Allowed cancer types:\n"
                + "\n".join(f"- {label}" for label in labels)
                + "\n\nPathology report:\n"
                + report_text
            ),
        },
    ]


def ajcc_stage_messages(report_text, stage_options=None):
    labels = stage_options or AJCC_STAGE_OPTIONS
    return [
        {
            "role": "system",
            "content": _json_system(
                'Output exactly this schema: {"ajcc_stage": "<one allowed label>"}'
            ),
        },
        {
            "role": "user",
            "content": (
                "Determine the high-level AJCC stage group for this pathology report. "
                "First look for an explicitly stated pathologic stage, AJCC stage, or stage group in the report; "
                "if a substage such as Stage IIA, IIIB, or IVA is stated, map it to Stage II, Stage III, or Stage IV. "
                "If no explicit stage group is stated, infer the best high-level stage group. "
                "Use the AJCC evidence present in the report, including tumor extent, "
                "nodal involvement, and distant metastasis where available. "
                "Allowed AJCC stage labels:\n"
                + "\n".join(f"- {label}" for label in labels)
                + "\n\nPathology report:\n"
                + report_text
            ),
        },
    ]


def prognosis_messages(report_text, cancer_type=None, mean_dss=None, examples=None):
    context = []
    if cancer_type:
        context.append(f"Cancer type: {cancer_type}")
    if mean_dss not in (None, ""):
        context.append(f"Mean disease-specific survival threshold: {mean_dss}")

    example_block = ""
    if examples:
        rendered = []
        for example in examples:
            label = example.get("label") or example.get("prognosis") or example.get("answer")
            summary = (
                example.get("summary")
                or example.get("report_summary")
                or example.get("report_text")
                or example.get("pathology_report")
                or example.get("text")
                or ""
            )
            rendered.append(
                "Example report summary:\n"
                f"{summary}\n"
                f'Example json answer: {{"prognosis": "{label}"}}'
            )
        example_block = "\n\nReference examples:\n" + "\n\n".join(rendered)

    return [
        {
            "role": "system",
            "content": _json_system(
                'Output exactly this schema: {"prognosis": "good"} or {"prognosis": "poor"}.'
            ),
        },
        {
            "role": "user",
            "content": (
                "Classify prognosis as good if the patient is likely to survive beyond "
                "the cancer-type mean disease-specific survival threshold; otherwise classify it as poor. "
                "This is a retrospective benchmark task, not clinical advice.\n"
                + ("\n".join(context) + "\n" if context else "")
                + example_block
                + "\n\nPathology report:\n"
                + report_text
            ),
        },
    ]


def build_messages(task, report_text, cancer_type=None, mean_dss=None, examples=None):
    if task == "cancer_type":
        return cancer_type_messages(report_text)
    if task == "ajcc_stage":
        return ajcc_stage_messages(report_text)
    if task == "prognosis":
        return prognosis_messages(
            report_text,
            cancer_type=cancer_type,
            mean_dss=mean_dss,
            examples=examples,
        )
    raise ValueError(f"Unknown task: {task}")
