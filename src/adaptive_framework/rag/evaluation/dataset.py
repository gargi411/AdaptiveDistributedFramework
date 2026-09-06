"""Ground-truth clinical evaluation queries and dataset loader for RAG evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from adaptive_framework.rag.evaluation.evaluation_case import RAGEvaluationCase

# 25 ground-truth clinical queries verified against the synthetic clinical notes dataset.
# Queries avoid data leakage (no document_ids in questions).
DEFAULT_GROUND_TRUTH_CASES: tuple[RAGEvaluationCase, ...] = (
    RAGEvaluationCase(
        case_id="eval_001",
        question="What diagnosis is documented for patient Soumya Das?",
        relevant_document_ids=("clinical_note_0001",),
        relevant_chunk_ids=("chk_synth_0001",),
        category="diagnosis",
        description="Ischemic stroke diagnosis for Soumya Das.",
    ),
    RAGEvaluationCase(
        case_id="eval_002",
        question="What did the EEG investigation reveal for patient Soumya Das?",
        relevant_document_ids=("clinical_note_0001",),
        relevant_chunk_ids=("chk_synth_0001",),
        category="investigations",
        description="EEG investigation showing abnormal spikes for Soumya Das.",
    ),
    RAGEvaluationCase(
        case_id="eval_003",
        question="What diagnosis and discharge medication were recorded for Anitha Rao?",
        relevant_document_ids=("clinical_note_0002",),
        relevant_chunk_ids=("chk_synth_0002",),
        category="treatment",
        description="Osteoarthritis diagnosis and Paracetamol medication for Anitha Rao.",
    ),
    RAGEvaluationCase(
        case_id="eval_004",
        question="What findings were seen on the spine MRI for Anitha Rao?",
        relevant_document_ids=("clinical_note_0002",),
        relevant_chunk_ids=("chk_synth_0002",),
        category="investigations",
        description="Spine MRI showing L4-L5 disc herniation for Anitha Rao.",
    ),
    RAGEvaluationCase(
        case_id="eval_005",
        question="What were the chief complaints and duration reported by patient Amit Sharma?",
        relevant_document_ids=("clinical_note_0003",),
        relevant_chunk_ids=("chk_synth_0003",),
        category="symptoms",
        description="Right knee pain for 30 months reported by Amit Sharma.",
    ),
    RAGEvaluationCase(
        case_id="eval_006",
        question="What diagnosis and CRP investigation results were documented for 45-year-old Ananya Banerjee?",
        relevant_document_ids=("clinical_note_0004",),
        relevant_chunk_ids=("chk_synth_0004",),
        category="diagnosis",
        description="Acute gastroenteritis with positive CRP for Ananya Banerjee.",
    ),
    RAGEvaluationCase(
        case_id="eval_007",
        question="What findings did endoscopy reveal for patient Rajesh Gupta?",
        relevant_document_ids=("clinical_note_0005",),
        relevant_chunk_ids=("chk_synth_0005",),
        category="investigations",
        description="Endoscopy confirming gastritis for Rajesh Gupta.",
    ),
    RAGEvaluationCase(
        case_id="eval_008",
        question="What discharge medication was prescribed to Rajesh Gupta for acute gastritis?",
        relevant_document_ids=("clinical_note_0005",),
        relevant_chunk_ids=("chk_synth_0005",),
        category="treatment",
        description="Amoxicillin 500mg TDS for Rajesh Gupta.",
    ),
    RAGEvaluationCase(
        case_id="eval_009",
        question="What diagnosis and CBC laboratory results were recorded for 75-year-old Ananya Banerjee?",
        relevant_document_ids=("clinical_note_0006",),
        relevant_chunk_ids=("chk_synth_0006",),
        category="diagnosis",
        description="Viral Exanthem with elevated WBC for 75-year-old Ananya Banerjee.",
    ),
    RAGEvaluationCase(
        case_id="eval_010",
        question="What symptoms and neurological presentation did Aman Verma present with?",
        relevant_document_ids=("clinical_note_0007",),
        relevant_chunk_ids=("chk_synth_0007",),
        category="symptoms",
        description="Right side weakness and speech difficulty in Aman Verma.",
    ),
    RAGEvaluationCase(
        case_id="eval_011",
        question="What diagnostic procedures confirmed breast carcinoma in Anjali Shah?",
        relevant_document_ids=("clinical_note_0008",),
        relevant_chunk_ids=("chk_synth_0008",),
        category="investigations",
        description="Mammogram and biopsy confirming malignancy in Anjali Shah.",
    ),
    RAGEvaluationCase(
        case_id="eval_012",
        question="What PET scan and CBC findings were documented for Suresh Iyer with breast carcinoma?",
        relevant_document_ids=("clinical_note_0009",),
        relevant_chunk_ids=("chk_synth_0009",),
        category="investigations",
        description="CBC anemia and PET scan metastasis in Suresh Iyer.",
    ),
    RAGEvaluationCase(
        case_id="eval_013",
        question="What did knee X-ray show and what was the diagnosis for Harpreet Kaur?",
        relevant_document_ids=("clinical_note_0010",),
        relevant_chunk_ids=("chk_synth_0010",),
        category="diagnosis",
        description="X-ray degenerative changes and osteoarthritis for Harpreet Kaur.",
    ),
    RAGEvaluationCase(
        case_id="eval_014",
        question="What symptoms and diagnosis were documented for pediatric patient Ravi Kumar?",
        relevant_document_ids=("clinical_note_0011",),
        relevant_chunk_ids=("chk_synth_0011",),
        category="symptoms",
        description="Fever and rash for 14 days, viral exanthem in Ravi Kumar.",
    ),
    RAGEvaluationCase(
        case_id="eval_015",
        question="What were the MRI and CT scan findings for stroke patient Gursharan Singh?",
        relevant_document_ids=("clinical_note_0013",),
        relevant_chunk_ids=("chk_synth_0013",),
        category="investigations",
        description="MRI left MCA infarct, CT no bleed in Gursharan Singh.",
    ),
    RAGEvaluationCase(
        case_id="eval_016",
        question="What diagnosis and discharge advice were given to Baby Priya?",
        relevant_document_ids=("clinical_note_0014",),
        relevant_chunk_ids=("chk_synth_0014",),
        category="treatment",
        description="Viral exanthem treated with Zinc syrup for Baby Priya.",
    ),
    RAGEvaluationCase(
        case_id="eval_017",
        question="What diagnosis was given to Rohan Desai who presented with knee pain?",
        relevant_document_ids=("clinical_note_0015",),
        relevant_chunk_ids=("chk_synth_0015",),
        category="diagnosis",
        description="Lumbar disc herniation diagnosed in Rohan Desai.",
    ),
    RAGEvaluationCase(
        case_id="eval_018",
        question="What clinical history and diagnosis were recorded for Priya Singh in oncology?",
        relevant_document_ids=("clinical_note_0016",),
        relevant_chunk_ids=("chk_synth_0016",),
        category="symptoms",
        description="Unexplained weight loss, fatigue, and breast carcinoma in Priya Singh.",
    ),
    RAGEvaluationCase(
        case_id="eval_019",
        question="What diagnosis and EEG results were noted for Suresh Iyer who suffered from severe headaches?",
        relevant_document_ids=("clinical_note_0018",),
        relevant_chunk_ids=("chk_synth_0018",),
        category="diagnosis",
        description="Migraine diagnosis with EEG abnormal spikes for Suresh Iyer.",
    ),
    RAGEvaluationCase(
        case_id="eval_020",
        question="What diagnosis of unknown primary was documented for Gursharan Singh?",
        relevant_document_ids=("clinical_note_0019",),
        relevant_chunk_ids=("chk_synth_0019",),
        category="diagnosis",
        description="Metastatic adenocarcinoma of unknown primary for Gursharan Singh.",
    ),
    RAGEvaluationCase(
        case_id="eval_021",
        question="What neurological diagnosis and follow-up advice were given to Mohan Lal?",
        relevant_document_ids=("clinical_note_0020",),
        relevant_chunk_ids=("chk_synth_0020",),
        category="treatment",
        description="Ischemic stroke with advice to avoid triggers for Mohan Lal.",
    ),
    RAGEvaluationCase(
        case_id="eval_022",
        question="What X-ray findings and antibiotic were documented for Geeta Devi?",
        relevant_document_ids=("clinical_note_022", "clinical_note_0022"),
        relevant_chunk_ids=("chk_synth_0022",),
        category="treatment",
        description="Pneumonia on X-ray and Amoxicillin for Geeta Devi.",
    ),
    RAGEvaluationCase(
        case_id="eval_023",
        question="What diagnosis and endoscopy investigation were recorded for Divya Krishnan presenting with abdominal pain?",
        relevant_document_ids=("clinical_note_0023",),
        relevant_chunk_ids=("chk_synth_0023",),
        category="investigations",
        description="Community-acquired pneumonia and gastritis on endoscopy for Divya Krishnan.",
    ),
    RAGEvaluationCase(
        case_id="eval_024",
        question="Which patients have discharge records documenting Carcinoma of the Breast?",
        relevant_document_ids=("clinical_note_0008", "clinical_note_0009", "clinical_note_0016"),
        relevant_chunk_ids=("chk_synth_0008", "chk_synth_0009", "chk_synth_0016"),
        category="multi_evidence",
        description="Multi-document query matching multiple breast carcinoma cases.",
    ),
    RAGEvaluationCase(
        case_id="eval_025",
        question="Which patients presented with Ischemic Stroke and had EEG showing abnormal spikes?",
        relevant_document_ids=("clinical_note_0001", "clinical_note_0007", "clinical_note_0020"),
        relevant_chunk_ids=("chk_synth_0001", "chk_synth_0007", "chk_synth_0020"),
        category="multi_evidence",
        description="Multi-document query matching multiple ischemic stroke cases with EEG spikes.",
    ),
)


def load_ground_truth_cases(filepath: Path | None = None) -> list[RAGEvaluationCase]:
    """Load ground-truth evaluation cases from JSON file or return defaults.

    Args:
        filepath: Optional path to JSON ground-truth file.

    Returns:
        List of RAGEvaluationCase objects.
    """
    if filepath and filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [
            RAGEvaluationCase(
                case_id=item["case_id"],
                question=item["question"],
                relevant_document_ids=tuple(item["relevant_document_ids"]),
                relevant_chunk_ids=tuple(item.get("relevant_chunk_ids", ())),
                category=item.get("category", "general"),
                description=item.get("description", ""),
            )
            for item in data
        ]

    return list(DEFAULT_GROUND_TRUTH_CASES)


def save_ground_truth_cases(cases: Sequence[RAGEvaluationCase], filepath: Path) -> None:
    """Save ground-truth cases to a JSON file.

    Args:
        cases: Sequence of RAGEvaluationCase objects.
        filepath: Target output path.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in cases], f, indent=2)
