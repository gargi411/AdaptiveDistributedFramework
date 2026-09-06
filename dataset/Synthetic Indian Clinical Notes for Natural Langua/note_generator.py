
import random
import json
import argparse
from typing import List, Dict, Any

# =====================================================================================
# DATA POOLS FOR RANDOMIZED GENERATION
# =====================================================================================

# Expanded list of Indian names for more diversity
indian_names = [
    'Rajesh Gupta', 'Aman Verma', 'Geeta Devi', 'Pooja Sharma', 'Ajay Singh', 'Veena Kumari',
    'Ravi Kumar', 'Sunita Devi', 'Amit Sharma', 'Priya Singh', 'Mohan Lal', 'Neha Patel',
    'Suresh Iyer', 'Anitha Rao', 'Vijay Kumar', 'Divya Krishnan', 'Rohan Desai', 'Anjali Shah',
    'Soumya Das', 'Ananya Banerjee', 'Gursharan Singh', 'Harpreet Kaur', 'Mohammed Ali', 'Fatima Begum',
    'Joseph D\'Souza', 'Maria Fernandes', 'Baby Aarav', 'Baby Priya',
]

# Expanded list of ages and sexes
ages_sexes = [
    '8 months/M', '6 months/F', '5 years/F', '17 years/M', '22/M', '28/F', '35/M', '45/F',
    '58/F', '62/M', '75/F',
]

# Expanded list of addresses to include more regions
addresses = [
    'Connaught Place, New Delhi', 'Bandra West, Mumbai', 'Koramangala, Bangalore',
    'Salt Lake City, Kolkata', 'Anna Nagar, Chennai', 'HITEC City, Hyderabad',
    'Jaipur, Rajasthan', 'Lucknow, Uttar Pradesh', 'Kaithal, Haryana', 'Varanasi, Uttar Pradesh',
    'Village Sisai, District Gumla, Jharkhand', 'Moga, Punjab'
]

# Expanded list of hospitals
hospitals = [
    'AIIMS Delhi', 'PGIMER Chandigarh', 'JIPMER Puducherry', 'Apollo Hospital Chennai',
    'Fortis Hospital Delhi', 'Max Super Speciality Hospital Delhi', 'Medanta The Medicity Gurgaon',
    'Safdarjung Hospital Delhi', 'Government Hospital Kaithal', 'Tata Memorial Hospital Mumbai',
]

# Expanded list of doctors
doctors = [
    'Dr. Anita Sharma', 'Dr. Vikram Singh', 'Dr. Neha Patel', 'Dr. Rajesh Kumar',
    'Dr. Arvind Subramanian (Cardiologist)', 'Dr. Anjali Desai (Pediatrician)',
    'Dr. Randeep Guleria (Pulmonologist)', 'Dr. Naresh Trehan (Cardiac Surgeon)',
]

# Unique identifiers
uhids = [str(random.randint(100000000, 999999999)) for _ in range(50)]
admission_nos = ['ADM-' + str(random.randint(20260000, 20269999)) for _ in range(50)]

# =====================================================================================
# SPECIALTY-SPECIFIC TEMPLATES
# =====================================================================================

specialties = {
    'cardiology': {
        'complaints': ['Chest pain aur saans phoolna for {days} days.', 'Palpitations and dizziness for {days} days.'],
        'hpi': ['Substernal pain on exertion, with diaphoresis.', 'Irregular heartbeat, no syncope.'],
        'past': ['Hypertension since {years} years, on meds.', 'Coronary artery disease, post-stent.'],
        'exam': ['BP: {bp}, Heart: S1/S2 normal, no murmur.', 'Pulse: Irregular.'],
        'invest': ['ECG: ST elevation; Trop-I: Positive; Echo: EF {ef}%; Angiogram: {block}% blockage in LAD.'],
        'diag': ['NSTEMI (ICD: I21.4).', 'Atrial Fibrillation (ICD: I48.91).'],
        'course': ['Admitted to CCU. Thrombolysis kiya gaya, then angioplasty with stent. Pain relieved.', 'Antiarrhythmics given, rhythm stabilized.'],
        'advice': ['Meds: Tab. Aspirin 75mg OD, Tab. Atorvastatin 40mg HS.', 'Diet: Low-salt; Follow-up: 7 days.']
    },
    'neurology': {
        'complaints': ['Right side weakness aur bolne mein difficulty for {days} days.', 'Severe headache with nausea for {days} days.'],
        'hpi': ['Sudden onset, with facial droop.', 'Migraine-like, photophobia.'],
        'past': ['Hypertension, smoker.', 'Epilepsy since childhood.'],
        'exam': ['Power: {power}/5 right limbs; Reflexes: Hyperactive.', 'Cranial nerves: Intact.'],
        'invest': ['MRI: Left MCA infarct; CT: No bleed.', 'EEG: Abnormal spikes.'],
        'diag': ['Ischemic Stroke (ICD: I63.9).', 'Migraine (ICD: G43.9).'],
        'course': ['IV thrombolysis, physiotherapy. Improved.', 'Antimigraine prophylaxis started.'],
        'advice': ['Meds: Tab. Clopidogrel 75mg OD.', 'Avoid triggers; Follow-up: 10 days.']
    },
    'pediatrics': {
        'complaints': ['Loose motions aur vomiting for {days} days.', 'Fever and rash for {days} days.'],
        'hpi': ['Acute, with mild dehydration.', 'High fever, measles-like rash.'],
        'past': ['Vaccinations up-to-date.', 'Recent URI.'],
        'exam': ['Dehydrated, abdomen soft.', 'Temp: 102°F, throat congested.'],
        'invest': ['CBC: WBC elevated; Stool: No parasites.', 'CRP: Positive.'],
        'diag': ['Acute Gastroenteritis (ICD: A09).', 'Viral Exanthem (ICD: B09).'],
        'course': ['ORS and IV fluids. Zinc given.', 'Symptomatic treatment, resolved.'],
        'advice': ['Meds: Syrup Zinc 5ml OD.', 'Diet: Breastfeed; Follow-up: 5 days.']
    },
    'general_medicine': {
        'complaints': ['Fever aur cough for {days} days.', 'Abdominal pain for {days} days.'],
        'hpi': ['Productive cough, no hemoptysis.', 'Epigastric, post-meal.'],
        'past': ['Diabetes on insulin.', 'GERD history.'],
        'exam': ['Temp: 102°F; Chest: Crepitations.', 'Abdomen: Tender epigastrium.'],
        'invest': ['CBC: WBC high; X-ray: Pneumonia.', 'Endoscopy: Gastritis.'],
        'diag': ['Community-Acquired Pneumonia (ICD: J18.9).', 'Acute Gastritis (ICD: K29.00).'],
        'course': ['IV antibiotics. Fever subsided.', 'PPI and antacids given.'],
        'advice': ['Meds: Tab. Amoxicillin 500mg TDS.', 'Diet: Bland; Follow-up: 1 week.']
    },
    'orthopaedics': {
        'complaints': ['Pain in right knee for {days} months.', 'Lower back pain radiating to the leg for {days} weeks.'],
        'hpi': ['Pain increases on walking and climbing stairs.', 'Pain is sharp and shooting in nature.'],
        'past': ['History of fall a few years back.', 'No history of chronic illness.'],
        'exam': ['Swelling and tenderness over the right knee joint.', 'Restricted range of motion of the lumbar spine.'],
        'invest': ['X-ray of the knee shows degenerative changes.', 'MRI of the spine shows disc herniation at L4-L5.'],
        'diag': ['Osteoarthritis of the right knee (ICD: M17.11).', 'Lumbar Disc Herniation (ICD: M51.26).'],
        'course': ['Advised for physiotherapy and lifestyle modifications.', 'Managed conservatively with analgesics and rest.'],
        'advice': ['Meds: Tab. Paracetamol 650mg SOS for pain.', 'Avoid lifting heavy weights; Follow-up after 2 weeks.']
    },
    'oncology': {
        'complaints': ['Unexplained weight loss and fatigue for {days} months.', 'Lump in the breast for {days} weeks.'],
        'hpi': ['Patient noticed a gradual loss of appetite.', 'The lump is painless and hard to touch.'],
        'past': ['Family history of cancer.', 'No history of major illness.'],
        'exam': ['Cachexic look, pallor present.', '2x2 cm hard lump in the upper outer quadrant of the left breast.'],
        'invest': ['CBC shows anemia. PET scan shows metastasis.', 'Mammogram and biopsy confirmed malignancy.'],
        'diag': ['Metastatic Adenocarcinoma of unknown primary (ICD: C80.0).', 'Carcinoma of the Breast (ICD: C50.9).'],
        'course': ['Patient started on palliative chemotherapy.', 'Underwent mastectomy followed by chemotherapy.'],
        'advice': ['High protein diet.', 'Regular follow-up in the oncology OPD.']
    }
}

# =====================================================================================
# NOTE GENERATION LOGIC
# =====================================================================================

length_configs = {
    'long': {'detail_multiplier': 2, 'add_sections': ['Family History: No significant history.', 'Allergies: None known.']},
    'medium': {'detail_multiplier': 1, 'add_sections': []},
    'short': {'detail_multiplier': 0.5, 'add_sections': []}
}

def generate_synthetic_note(specialty: str, length_option: str = 'medium') -> str:
    """
    Generates a single synthetic clinical note based on specialty and length.

    Args:
        specialty: The medical specialty for the note.
        length_option: 'long', 'medium', or 'short'.

    Returns:
        A formatted string representing the synthetic clinical note.
    """
    config = length_configs.get(length_option.lower(), length_configs['medium'])
    mul = config['detail_multiplier']

    # Random fills for placeholders
    days = random.randint(2, 30)
    years = random.randint(3, 10)
    bp = f'{random.randint(120, 180)}/{random.randint(80, 110)} mmHg'
    power = random.randint(3, 4)
    ef = random.randint(45, 60)
    block = random.randint(50, 90)

    spec_data = specialties[specialty]

    note_parts = {
        'hospital': random.choice(hospitals),
        'name': random.choice(indian_names),
        'age_sex': random.choice(ages_sexes),
        'uhid': random.choice(uhids),
        'adm_no': random.choice(admission_nos),
        'address': random.choice(addresses),
        'adm_date': f'0{random.randint(1, 9)}/02/2026 {random.randint(8, 12):02}:00 AM',
        'dis_date': f'1{random.randint(0, 5)}/02/2026 {random.randint(1, 3):02}:00 PM',
        'doctor': random.choice(doctors),
        'complaints': random.choice(spec_data['complaints']).format(days=days),
        'hpi': ' '.join([random.choice(spec_data['hpi'])] * int(mul) if mul > 0 else [random.choice(spec_data['hpi'])]), 
        'past': random.choice(spec_data['past']).format(years=years),
        'exam': random.choice(spec_data['exam']).format(bp=bp, power=power),
        'invest': random.choice(spec_data['invest']).format(ef=ef, block=block),
        'diag': random.choice(spec_data['diag']),
        'course': ' '.join([random.choice(spec_data['course'])] * int(mul) if mul > 0 else [random.choice(spec_data['course'])]), 
        'advice': '\n- '.join([random.choice(spec_data['advice'])] * int(mul) if mul > 0 else [random.choice(spec_data['advice'])]) + '\n' + '\n'.join(config['add_sections'])
    }

    # Assemble the note
    note = f"""
{note_parts['hospital']}
Discharge Summary

Patient Name: {note_parts['name']}
Age/Sex: {note_parts['age_sex']}
UHID: {note_parts['uhid']}
Admission No: {note_parts['adm_no']}
Address: {note_parts['address']}
Admission Date: {note_parts['adm_date']}
Discharge Date: {note_parts['dis_date']}
Treating Consultant: {note_parts['doctor']}, {specialty.capitalize()}

Chief Complaints: {note_parts['complaints']}

History of Present Illness: {note_parts['hpi']} Patient ko {random.choice(['fever tha', 'dard tha', 'weakness feel kar raha tha'])}.

Past History: {note_parts['past']}

Physical Examination: {note_parts['exam']}
- PICCKLE: Pallor: +, Icterus: -, Cyanosis: -, Clubbing: -, Koilonychia: -, Lymphadenopathy: -, Edema: -

Investigations: {note_parts['invest']}

Diagnosis: {note_parts['diag']}

Course in Hospital: {note_parts['course']} No major complications, but monitored for {random.choice(['dengue-like symptoms', 'malaria risk', 'post-viral fatigue'])}.

Socio-Economic Status: Class {random.choice(['II', 'III', 'IV'])} as per modified Kuppuswamy scale.
Traditional Medicine History: Used {random.choice(['Tulsi for cough', 'Haldi milk for pain', 'none'])}.

Condition at Discharge: Stable, afebrile.

Advice on Discharge:
- {note_parts['advice']}

{random.choice(doctors)}
Signature
"""
    return note.strip()

def generate_clinical_notes(x: int, y: int, length_option: str) -> List[Dict[str, Any]]:
    """
    Generates a list of synthetic clinical notes.

    Args:
        x: The total number of notes to generate.
        y: The number of different medical specialties to choose from.
        length_option: 'long', 'medium', or 'short'.

    Returns:
        A list of dictionaries, where each dictionary represents a note.
    """
    if y > len(specialties):
        raise ValueError(f"Only {len(specialties)} specialties available: {list(specialties.keys())}")

    selected_specialties = random.sample(list(specialties.keys()), y)
    notes = []

    for i in range(x):
        spec = random.choice(selected_specialties)
        note_content = generate_synthetic_note(spec, length_option)
        notes.append({'id': i + 1, 'specialty': spec, 'length': length_option, 'note': note_content})

    return notes

# =====================================================================================
# CLI INTERFACE
# =====================================================================================

def main():
    """Main function to run the CLI."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic Indian clinical notes for NLP research.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-x', '--num_notes',
        type=int,
        required=True,
        help="Total number of clinical notes to generate."
    )
    parser.add_argument(
        '-y', '--num_types',
        type=int,
        required=True,
        help="Number of different medical specialties (types) to include."
    )
    parser.add_argument(
        '-l', '--length',
        type=str,
        choices=['long', 'medium', 'short'],
        default='medium',
        help="Length of the generated notes."
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='synthetic_notes.json',
        help="Output file name to save the notes (e.g., synthetic_notes.json)."
    )

    args = parser.parse_args()

    try:
        notes = generate_clinical_notes(args.num_notes, args.num_types, args.length)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(notes, f, ensure_ascii=False, indent=2)
        print(f"Successfully generated {args.num_notes} notes and saved to {args.output}")
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
