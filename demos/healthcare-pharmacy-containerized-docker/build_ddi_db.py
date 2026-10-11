#!/usr/bin/env python3
"""
Build DDI ChromaDB — Drug-Drug Interaction vector database for the Pharmacy demo.
Downloads DDInter 1.0 bulk CSV dataset (A-Z files), falls back to curated seed dataset,
embeds via Ollama nomic-embed-text, stores in ChromaDB.
"""

import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── 1. Install dependencies if missing ──────────────────────────────────────

REQUIRED = ["chromadb", "requests", "pandas"]
missing = [pkg for pkg in REQUIRED if importlib.util.find_spec(pkg) is None]
if missing:
    print(f"Installing missing packages: {', '.join(missing)}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install"] + missing + ["--break-system-packages", "-q"]
    )

import chromadb
import pandas as pd
import requests

# ── Config ───────────────────────────────────────────────────────────────────

DDI_BASE_URL = "https://ddinter.scbdd.com/static/media/download/"
DDI_FILES = [f"ddinter_downloads_code_{chr(i)}.csv" for i in range(ord('A'), ord('Z') + 1)]
DOWNLOAD_DIR = Path("/tmp/ddinter1_download")
DB_DIR = Path.home() / "demos" / "healthcare-pharmacy-containerized-docker" / "ddi_chromadb"
COLLECTION_NAME = "ddi_interactions"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 100

SEVERITY_MAP = {
    "minor": "Minor", "moderate": "Moderate", "major": "Major",
    "contraindicated": "Contraindicated",
}


# ── 2. Download DDInter 1.0 bulk CSVs or fall back to seed dataset ──────────

def download_ddinter() -> list[dict]:
    """Download DDInter 1.0 A-Z CSV files and parse into records."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    downloaded = 0

    print(f"Downloading DDInter 1.0 from {DDI_BASE_URL} ...")
    for fname in DDI_FILES:
        url = DDI_BASE_URL + fname
        csv_path = DOWNLOAD_DIR / fname
        letter = fname.split("_")[-1].split(".")[0]
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                print(f"  [{letter}] 404 — skipping")
                continue
            resp.raise_for_status()
            with open(csv_path, "wb") as f:
                f.write(resp.content)
            df = pd.read_csv(csv_path, on_bad_lines="skip", low_memory=False)
            frames.append(df)
            downloaded += 1
            print(f"  [{letter}] {len(df):,} records ({len(resp.content) / 1024:.0f} KB)")
        except Exception as e:
            print(f"  [{letter}] Failed: {e}")
            continue

    if not frames:
        return []

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nDownloaded {downloaded}/26 files — {len(combined):,} total rows")
    print(f"Columns: {list(combined.columns)}")

    # DDInter 1.0 columns: DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level
    # Some files may also have: A, B, Mechanism — handle both schemas
    drug_a_col = "Drug_A" if "Drug_A" in combined.columns else "A"
    drug_b_col = "Drug_B" if "Drug_B" in combined.columns else "B"
    mech_col = "Mechanism" if "Mechanism" in combined.columns else None

    records = []
    for _, row in combined.iterrows():
        da = str(row.get(drug_a_col, "")).strip()
        db = str(row.get(drug_b_col, "")).strip()
        if not da or not db or da == "nan" or db == "nan":
            continue

        raw_sev = str(row.get("Level", "")).strip().lower()
        severity = SEVERITY_MAP.get(raw_sev, "Moderate")
        mechanism = ""
        if mech_col:
            mechanism = str(row.get(mech_col, "")).strip()
            if mechanism == "nan":
                mechanism = ""

        records.append({
            "drug_a": da.lower(),
            "drug_b": db.lower(),
            "severity": severity,
            "mechanism": mechanism or "See clinical references",
            "clinical_significance": "",
            "management": "",
            "alternatives": "",
        })

    print(f"Parsed {len(records):,} valid DDI records")
    return records


def build_seed_dataset() -> list[dict]:
    """Curated 50-pair seed dataset covering major DDI categories."""
    print("Building curated 50-pair seed dataset ...")
    return [
        # ── MAOIs + SSRIs (serotonin syndrome) ───────────────────────────
        {"drug_a": "Phenelzine", "drug_b": "Fluoxetine", "severity": "Contraindicated",
         "mechanism": "Combined MAO inhibition and serotonin reuptake inhibition causes excessive serotonergic activity",
         "clinical_significance": "Risk of fatal serotonin syndrome: hyperthermia, rigidity, myoclonus, autonomic instability",
         "management": "Do not co-administer. Allow 5-week washout after fluoxetine, 2 weeks after phenelzine",
         "alternatives": "Bupropion (with caution), mirtazapine (with caution), CBT"},
        {"drug_a": "Tranylcypromine", "drug_b": "Sertraline", "severity": "Contraindicated",
         "mechanism": "Irreversible MAO inhibition plus SSRI increases serotonin to toxic levels",
         "clinical_significance": "Life-threatening serotonin syndrome",
         "management": "Absolutely contraindicated. 2-week washout required between agents",
         "alternatives": "Bupropion, nortriptyline (with caution)"},
        {"drug_a": "Isocarboxazid", "drug_b": "Paroxetine", "severity": "Contraindicated",
         "mechanism": "Dual serotonergic potentiation via MAO-A inhibition and serotonin reuptake blockade",
         "clinical_significance": "Serotonin syndrome risk — potentially fatal",
         "management": "Never combine. Minimum 14-day MAOI washout, 2-week SSRI washout",
         "alternatives": "Bupropion, tricyclics (selected, with monitoring)"},
        {"drug_a": "Phenelzine", "drug_b": "Venlafaxine", "severity": "Contraindicated",
         "mechanism": "SNRI serotonin/norepinephrine reuptake inhibition combined with MAO inhibition",
         "clinical_significance": "Serotonin syndrome and hypertensive crisis risk",
         "management": "Contraindicated. 2-week washout from MAOI, 1 week from venlafaxine",
         "alternatives": "Bupropion, mirtazapine"},
        {"drug_a": "Selegiline", "drug_b": "Citalopram", "severity": "Contraindicated",
         "mechanism": "Even selective MAO-B inhibitors lose selectivity at higher doses; combined with SSRI increases serotonin",
         "clinical_significance": "Serotonin syndrome, especially at selegiline doses >10mg/day",
         "management": "Avoid combination. Use transdermal selegiline with extreme caution only",
         "alternatives": "Rasagiline dose-limited monotherapy, bupropion"},

        # ── Benzodiazepines + Opioids (respiratory depression) ───────────
        {"drug_a": "Alprazolam", "drug_b": "Oxycodone", "severity": "Major",
         "mechanism": "Additive CNS and respiratory depression via GABA-A and mu-opioid receptor agonism",
         "clinical_significance": "FDA black box warning — increased risk of sedation, respiratory depression, coma, death",
         "management": "Avoid if possible. If essential, use lowest effective doses and shortest duration. Monitor respiratory rate and sedation",
         "alternatives": "Non-benzodiazepine anxiolytics (buspirone, hydroxyzine); non-opioid analgesics (NSAIDs, acetaminophen)"},
        {"drug_a": "Diazepam", "drug_b": "Morphine", "severity": "Major",
         "mechanism": "Synergistic CNS depression — both enhance inhibitory neurotransmission",
         "clinical_significance": "Profound sedation, respiratory failure, death. Risk amplified in elderly and hepatically impaired",
         "management": "Limit to cases where alternatives inadequate. Reduce opioid dose by 50%. Continuous pulse oximetry",
         "alternatives": "Gabapentin for pain, buspirone for anxiety"},
        {"drug_a": "Lorazepam", "drug_b": "Fentanyl", "severity": "Major",
         "mechanism": "Combined GABA-ergic and opioid receptor-mediated respiratory depression",
         "clinical_significance": "Highest-risk combination due to fentanyl potency. Respiratory arrest reported even at therapeutic doses",
         "management": "Reserve for procedural sedation with full monitoring. Naloxone and flumazenil at bedside",
         "alternatives": "Dexmedetomidine for procedural sedation, ketamine"},
        {"drug_a": "Clonazepam", "drug_b": "Hydrocodone", "severity": "Major",
         "mechanism": "Additive CNS/respiratory depression",
         "clinical_significance": "Increased overdose risk, particularly in opioid-naive patients",
         "management": "If unavoidable, start opioid at 50% dose. Patient education on sedation signs. Prescribe naloxone rescue",
         "alternatives": "Pregabalin, duloxetine for pain; hydroxyzine for anxiety"},
        {"drug_a": "Midazolam", "drug_b": "Methadone", "severity": "Major",
         "mechanism": "Dual CNS depression compounded by methadone's long half-life and QTc effects",
         "clinical_significance": "Prolonged respiratory depression, QTc prolongation, torsades de pointes risk",
         "management": "Avoid. If required for procedures, continuous cardiac and respiratory monitoring for 24+ hours",
         "alternatives": "Propofol for procedural sedation, buprenorphine for opioid maintenance"},

        # ── Warfarin interactions ────────────────────────────────────────
        {"drug_a": "Warfarin", "drug_b": "Aspirin", "severity": "Major",
         "mechanism": "Aspirin inhibits platelet aggregation and may displace warfarin from albumin binding; additive bleeding risk",
         "clinical_significance": "2-3x increased risk of major GI and intracranial hemorrhage",
         "management": "Avoid unless compelling indication (mechanical valve + CAD). Monitor INR closely. Add PPI for GI protection",
         "alternatives": "Clopidogrel if antiplatelet needed (still increases risk but less GI bleeding)"},
        {"drug_a": "Warfarin", "drug_b": "Amiodarone", "severity": "Major",
         "mechanism": "Amiodarone inhibits CYP2C9 and CYP3A4, reducing warfarin clearance by 30-50%",
         "clinical_significance": "INR may double within 1-2 weeks. Hemorrhage risk persists months after amiodarone discontinued (long t1/2)",
         "management": "Reduce warfarin dose by 30-50% when starting amiodarone. Check INR weekly for 4-6 weeks, then biweekly",
         "alternatives": "Dronedarone (less CYP inhibition), DOAC if AF indication"},
        {"drug_a": "Warfarin", "drug_b": "Fluconazole", "severity": "Major",
         "mechanism": "Fluconazole is a potent CYP2C9 inhibitor, markedly reducing S-warfarin metabolism",
         "clinical_significance": "INR elevation within 2-3 days. Serious bleeding events reported",
         "management": "Reduce warfarin dose by 25-50%. Monitor INR at day 3 and 7 of fluconazole therapy",
         "alternatives": "Terbinafine for dermatophytes, micafungin for systemic (no CYP2C9 effect)"},
        {"drug_a": "Warfarin", "drug_b": "Rifampin", "severity": "Major",
         "mechanism": "Rifampin is a potent CYP inducer (1A2, 2C9, 3A4) — accelerates warfarin metabolism dramatically",
         "clinical_significance": "INR drops to subtherapeutic within days. Thromboembolic events (stroke, PE) reported",
         "management": "Avoid combination. If essential, increase warfarin dose 2-3x with twice-weekly INR monitoring. Adjust on rifampin discontinuation",
         "alternatives": "Rifabutin (weaker inducer), DOAC with dose adjustment"},
        {"drug_a": "Warfarin", "drug_b": "Metronidazole", "severity": "Major",
         "mechanism": "Metronidazole inhibits CYP2C9, decreasing S-warfarin clearance",
         "clinical_significance": "INR elevation, bleeding risk — especially with courses >7 days",
         "management": "Monitor INR at day 3-5. Reduce warfarin by 25% for courses >5 days",
         "alternatives": "Vancomycin (oral for C. diff), tinidazole (shorter course)"},
        {"drug_a": "Warfarin", "drug_b": "Ibuprofen", "severity": "Major",
         "mechanism": "NSAIDs inhibit COX-1 platelet function and cause GI mucosal injury; displacement from protein binding",
         "clinical_significance": "3-6x increased risk of GI hemorrhage compared to warfarin alone",
         "management": "Avoid NSAIDs if possible. If needed, use lowest dose shortest duration with PPI. Monitor INR",
         "alternatives": "Acetaminophen (first-line), topical NSAIDs, tramadol with caution"},
        {"drug_a": "Warfarin", "drug_b": "Sulfamethoxazole-Trimethoprim", "severity": "Major",
         "mechanism": "Sulfamethoxazole inhibits CYP2C9; trimethoprim may reduce folate-dependent clotting factor synthesis",
         "clinical_significance": "INR can increase significantly within 3-5 days. Hemorrhagic events documented",
         "management": "Reduce warfarin by 10-25%. Check INR at day 3 and after course completion",
         "alternatives": "Amoxicillin-clavulanate, nitrofurantoin for UTI"},

        # ── Statins + CYP3A4 inhibitors (rhabdomyolysis) ────────────────
        {"drug_a": "Simvastatin", "drug_b": "Clarithromycin", "severity": "Contraindicated",
         "mechanism": "Clarithromycin is a strong CYP3A4 inhibitor, increasing simvastatin AUC 10-fold",
         "clinical_significance": "Severe rhabdomyolysis, acute kidney injury, death reported",
         "management": "Contraindicated. Suspend simvastatin during clarithromycin course",
         "alternatives": "Azithromycin (no CYP3A4 effect) for macrolide; rosuvastatin/pravastatin (not CYP3A4 metabolized) for statin"},
        {"drug_a": "Lovastatin", "drug_b": "Itraconazole", "severity": "Contraindicated",
         "mechanism": "Itraconazole potently inhibits CYP3A4 and P-glycoprotein, massively increasing lovastatin exposure",
         "clinical_significance": "Rhabdomyolysis with renal failure. FDA contraindication",
         "management": "Do not co-administer. Withhold lovastatin during azole antifungal therapy",
         "alternatives": "Fluconazole (lower CYP3A4 effect) or terbinafine; pravastatin or rosuvastatin for statin"},
        {"drug_a": "Atorvastatin", "drug_b": "Ritonavir", "severity": "Major",
         "mechanism": "Ritonavir is a potent CYP3A4 inhibitor, increasing atorvastatin levels 5-9 fold",
         "clinical_significance": "High risk of myopathy and rhabdomyolysis at standard atorvastatin doses",
         "management": "Use lowest atorvastatin dose (10mg). Monitor CK levels. Patient education on muscle pain symptoms",
         "alternatives": "Pravastatin or rosuvastatin (preferred with protease inhibitors), pitavastatin"},
        {"drug_a": "Simvastatin", "drug_b": "Diltiazem", "severity": "Major",
         "mechanism": "Diltiazem moderately inhibits CYP3A4, increasing simvastatin exposure 4-5 fold",
         "clinical_significance": "Increased myopathy risk. FDA limits simvastatin to 10mg with diltiazem",
         "management": "Do not exceed simvastatin 10mg/day. Consider statin switch",
         "alternatives": "Pravastatin, rosuvastatin, or pitavastatin (not CYP3A4 dependent)"},
        {"drug_a": "Simvastatin", "drug_b": "Amiodarone", "severity": "Major",
         "mechanism": "Amiodarone inhibits CYP3A4 and P-glycoprotein, increasing simvastatin levels",
         "clinical_significance": "Rhabdomyolysis risk. FDA limits simvastatin to 20mg with amiodarone",
         "management": "Do not exceed simvastatin 20mg/day with amiodarone. Monitor CK if muscle symptoms",
         "alternatives": "Pravastatin, rosuvastatin (preferred)"},
        {"drug_a": "Atorvastatin", "drug_b": "Cyclosporine", "severity": "Contraindicated",
         "mechanism": "Cyclosporine inhibits CYP3A4, P-gp, and OATP1B1, dramatically increasing statin exposure",
         "clinical_significance": "Severe rhabdomyolysis. All statins affected but CYP3A4-metabolized statins worst",
         "management": "Avoid atorvastatin. If statin needed, use pravastatin 20mg or fluvastatin 40mg with CK monitoring",
         "alternatives": "Pravastatin (lowest interaction risk with cyclosporine), ezetimibe"},

        # ── Common contraindicated / high-severity pairs ─────────────────
        {"drug_a": "Methotrexate", "drug_b": "Trimethoprim", "severity": "Contraindicated",
         "mechanism": "Both are folate antagonists. Trimethoprim also reduces renal clearance of methotrexate",
         "clinical_significance": "Pancytopenia, severe mucositis, death — especially with high-dose methotrexate",
         "management": "Avoid combination. Use alternative antibiotics",
         "alternatives": "Amoxicillin, cephalexin, nitrofurantoin for UTI"},
        {"drug_a": "Sildenafil", "drug_b": "Nitroglycerin", "severity": "Contraindicated",
         "mechanism": "PDE5 inhibitors potentiate nitric oxide-mediated vasodilation from nitrates",
         "clinical_significance": "Profound refractory hypotension, myocardial infarction, death",
         "management": "Absolutely contraindicated. 24h washout for sildenafil, 48h for tadalafil before nitrate administration",
         "alternatives": "Ranolazine for angina (no nitrate interaction); alprostadil for ED"},
        {"drug_a": "Potassium Chloride", "drug_b": "Spironolactone", "severity": "Major",
         "mechanism": "Spironolactone is a potassium-sparing diuretic; exogenous potassium adds to retention",
         "clinical_significance": "Life-threatening hyperkalemia — cardiac arrhythmias, cardiac arrest",
         "management": "Avoid routine potassium supplementation with spironolactone. Monitor K+ within 1 week of initiation and after dose changes",
         "alternatives": "Loop diuretic if diuresis needed without K+ retention; dietary potassium counseling"},
        {"drug_a": "Lithium", "drug_b": "Hydrochlorothiazide", "severity": "Major",
         "mechanism": "Thiazides reduce renal lithium clearance by increasing proximal tubule reabsorption (sodium depletion)",
         "clinical_significance": "Lithium toxicity — tremor, ataxia, renal failure, seizures. Levels can rise 25-40%",
         "management": "If unavoidable, reduce lithium dose by 25-50%. Monitor lithium levels weekly for first month, then monthly",
         "alternatives": "Loop diuretics (less effect on lithium), amiloride (may actually stabilize lithium levels)"},
        {"drug_a": "Digoxin", "drug_b": "Amiodarone", "severity": "Major",
         "mechanism": "Amiodarone inhibits P-glycoprotein and reduces renal/non-renal digoxin clearance, increasing levels 70-100%",
         "clinical_significance": "Digoxin toxicity — nausea, visual disturbances, fatal cardiac arrhythmias",
         "management": "Reduce digoxin dose by 50% when starting amiodarone. Monitor digoxin levels at 1 and 2 weeks",
         "alternatives": "Beta-blocker for rate control (metoprolol), diltiazem if no HF"},
        {"drug_a": "Clopidogrel", "drug_b": "Omeprazole", "severity": "Major",
         "mechanism": "Omeprazole inhibits CYP2C19, blocking conversion of clopidogrel prodrug to active metabolite",
         "clinical_significance": "Reduced antiplatelet effect — increased risk of stent thrombosis and cardiovascular events",
         "management": "Avoid omeprazole and esomeprazole specifically. Use pantoprazole (minimal CYP2C19 inhibition)",
         "alternatives": "Pantoprazole, famotidine (H2 blocker)"},
        {"drug_a": "Ciprofloxacin", "drug_b": "Tizanidine", "severity": "Contraindicated",
         "mechanism": "Ciprofloxacin strongly inhibits CYP1A2, increasing tizanidine AUC 10-fold and Cmax 7-fold",
         "clinical_significance": "Severe hypotension, excessive sedation, respiratory depression",
         "management": "Contraindicated by FDA. Use alternative antibiotic or muscle relaxant",
         "alternatives": "Levofloxacin (weak CYP1A2 effect) or non-fluoroquinolone; baclofen or cyclobenzaprine for spasticity"},
        {"drug_a": "Linezolid", "drug_b": "Meperidine", "severity": "Contraindicated",
         "mechanism": "Linezolid is a reversible MAO inhibitor; meperidine has serotonergic activity",
         "clinical_significance": "Serotonin syndrome — can be fatal. Also risk with other opioids but highest with meperidine",
         "management": "Do not co-administer. If linezolid essential, use non-serotonergic opioid",
         "alternatives": "Vancomycin or daptomycin for MRSA; morphine or hydromorphone for pain (lower serotonin risk)"},
        {"drug_a": "Ergotamine", "drug_b": "Ritonavir", "severity": "Contraindicated",
         "mechanism": "Ritonavir potently inhibits CYP3A4, preventing ergotamine metabolism — massive ergot accumulation",
         "clinical_significance": "Acute ergotism — vasospasm, limb ischemia, gangrene, death",
         "management": "Absolutely contraindicated. No dose adjustment makes this safe",
         "alternatives": "Triptans (sumatriptan) with caution, gepants (ubrogepant — but also CYP3A4 substrate, use rimegepant)"},
        {"drug_a": "Carbamazepine", "drug_b": "Voriconazole", "severity": "Contraindicated",
         "mechanism": "Carbamazepine induces CYP2C19/3A4 reducing voriconazole to undetectable levels; voriconazole inhibits carbamazepine metabolism",
         "clinical_significance": "Bidirectional: treatment failure of fungal infection AND carbamazepine toxicity",
         "management": "Contraindicated. Use alternative antifungal or anticonvulsant",
         "alternatives": "Amphotericin B or micafungin; levetiracetam or lamotrigine for seizures"},
        {"drug_a": "Dofetilide", "drug_b": "Verapamil", "severity": "Contraindicated",
         "mechanism": "Verapamil inhibits renal cation transport (OCT2), increasing dofetilide levels by 50%",
         "clinical_significance": "QTc prolongation, torsades de pointes, sudden cardiac death",
         "management": "Contraindicated per FDA label. No dose adjustment overcomes risk",
         "alternatives": "Amiodarone (different mechanism), sotalol with QTc monitoring; amlodipine for calcium channel blockade"},
        {"drug_a": "Colchicine", "drug_b": "Clarithromycin", "severity": "Contraindicated",
         "mechanism": "Clarithromycin inhibits CYP3A4 and P-glycoprotein, preventing colchicine elimination",
         "clinical_significance": "Fatal colchicine toxicity — multi-organ failure, pancytopenia. Deaths reported even at standard doses",
         "management": "Do not co-administer in renal or hepatic impairment. In normal function: colchicine 0.3mg single dose, no repeat for 3 days",
         "alternatives": "Azithromycin (weak CYP3A4 effect); NSAIDs or corticosteroids for acute gout"},

        # ── Additional high-impact pairs ─────────────────────────────────
        {"drug_a": "metformin", "drug_b": "contrast dye", "severity": "High",
         "mechanism": "Iodinated contrast agents can cause contrast-induced nephropathy (acute kidney injury), reducing metformin renal clearance. Metformin accumulation leads to lactic acidosis, not hypoglycemia.",
         "clinical_significance": "Risk of potentially fatal lactic acidosis if metformin is not held before contrast administration. Standard protocol: hold metformin 48 hours before and after contrast, restart only after confirming renal function is stable.",
         "management": "Hold metformin 48 hours prior to contrast procedure. Check serum creatinine and eGFR before restarting. Monitor for signs of lactic acidosis: nausea, vomiting, abdominal pain, hyperventilation.",
         "alternatives": "Insulin for temporary glycemic control during metformin hold period. Use iso-osmolar or low-osmolar contrast agents to minimize nephrotoxicity risk."},
        {"drug_a": "Theophylline", "drug_b": "Ciprofloxacin", "severity": "Major",
         "mechanism": "Ciprofloxacin inhibits CYP1A2, the primary metabolic pathway for theophylline",
         "clinical_significance": "Theophylline toxicity — seizures, cardiac arrhythmias, nausea. Levels can double",
         "management": "Reduce theophylline dose by 30-50%. Monitor serum levels at day 2-3. Watch for toxicity signs",
         "alternatives": "Levofloxacin (minimal CYP1A2 effect), azithromycin, or amoxicillin-clavulanate"},
        {"drug_a": "Fluoxetine", "drug_b": "Tramadol", "severity": "Major",
         "mechanism": "Fluoxetine inhibits CYP2D6 (reducing tramadol activation) and adds serotonergic activity",
         "clinical_significance": "Dual risk: serotonin syndrome AND reduced analgesic efficacy",
         "management": "Avoid combination. If needed, use lowest tramadol dose with serotonin syndrome monitoring",
         "alternatives": "Acetaminophen, NSAIDs, or gabapentin for pain; if opioid needed, morphine (less serotonergic)"},
        {"drug_a": "ACE Inhibitor", "drug_b": "Potassium-Sparing Diuretic", "severity": "Major",
         "mechanism": "ACE inhibitors reduce aldosterone, decreasing potassium excretion; combined with K+-sparing diuretic compounds effect",
         "clinical_significance": "Hyperkalemia — cardiac conduction abnormalities, cardiac arrest. Risk higher in CKD, diabetes, elderly",
         "management": "Monitor K+ within 1 week and after any dose change. Avoid if eGFR <30. Dietary K+ counseling",
         "alternatives": "Loop diuretic or thiazide if diuresis needed; ARB has similar K+ risk"},
        {"drug_a": "Quetiapine", "drug_b": "Ketoconazole", "severity": "Major",
         "mechanism": "Ketoconazole inhibits CYP3A4, increasing quetiapine levels 6-fold",
         "clinical_significance": "Excessive sedation, orthostatic hypotension, QTc prolongation, neuroleptic malignant syndrome risk",
         "management": "Reduce quetiapine dose to 1/6th. Monitor QTc. Avoid if possible",
         "alternatives": "Fluconazole (weaker inhibitor) or topical antifungal; olanzapine (CYP1A2, not 3A4) for antipsychotic"},
        {"drug_a": "Allopurinol", "drug_b": "Azathioprine", "severity": "Contraindicated",
         "mechanism": "Allopurinol inhibits xanthine oxidase which metabolizes azathioprine/6-MP, causing 3-5x accumulation",
         "clinical_significance": "Severe pancytopenia, life-threatening myelosuppression",
         "management": "If unavoidable, reduce azathioprine dose to 25-33% of standard. Weekly CBC for 8 weeks then monthly",
         "alternatives": "Febuxostat (similar risk — also XO inhibitor); mycophenolate as immunosuppressant alternative"},
        {"drug_a": "Citalopram", "drug_b": "Ondansetron", "severity": "Major",
         "mechanism": "Both drugs independently prolong QTc interval via different ion channel effects",
         "clinical_significance": "Additive QTc prolongation — risk of torsades de pointes, syncope, sudden death",
         "management": "Use alternative antiemetic. If essential, obtain baseline ECG, monitor QTc, keep <500ms",
         "alternatives": "Metoclopramide, prochlorperazine, or dexamethasone for nausea"},
        {"drug_a": "Methotrexate", "drug_b": "NSAIDs", "severity": "Major",
         "mechanism": "NSAIDs reduce renal blood flow and tubular secretion of methotrexate, increasing levels",
         "clinical_significance": "Methotrexate toxicity — myelosuppression, mucositis, renal failure. Risk highest with high-dose MTX",
         "management": "Avoid NSAIDs 2 days before through 2 days after high-dose MTX. Low-dose MTX: monitor CBC, use with caution",
         "alternatives": "Acetaminophen for analgesia, corticosteroid injection for focal inflammation"},
        {"drug_a": "Pimozide", "drug_b": "Ketoconazole", "severity": "Contraindicated",
         "mechanism": "Ketoconazole inhibits CYP3A4, markedly increasing pimozide levels",
         "clinical_significance": "QTc prolongation, torsades de pointes, sudden cardiac death",
         "management": "Absolutely contraindicated. No safe dose combination exists",
         "alternatives": "Topical antifungal or terbinafine; aripiprazole for antipsychotic (lower QTc risk)"},
        {"drug_a": "Clozapine", "drug_b": "Carbamazepine", "severity": "Contraindicated",
         "mechanism": "Both cause bone marrow suppression independently; carbamazepine also induces CYP1A2/3A4 reducing clozapine levels",
         "clinical_significance": "Additive agranulocytosis risk (both carry independent black box warnings) plus subtherapeutic clozapine",
         "management": "Contraindicated. Do not co-administer under any circumstances",
         "alternatives": "Valproate or lamotrigine for mood stabilization with clozapine"},
        {"drug_a": "Tamoxifen", "drug_b": "Paroxetine", "severity": "Major",
         "mechanism": "Paroxetine is a potent CYP2D6 inhibitor, blocking conversion of tamoxifen to active endoxifen",
         "clinical_significance": "Reduced tamoxifen efficacy — 25-65% reduction in endoxifen levels. Increased breast cancer recurrence",
         "management": "Avoid all strong CYP2D6 inhibitors with tamoxifen",
         "alternatives": "Venlafaxine, citalopram, or escitalopram (weak CYP2D6 inhibitors) for depression/hot flashes"},

        # ── Benzodiazepine + Opioid expanded pairs (FDA black box) ───────
        {"drug_a": "Clonazepam", "drug_b": "Fentanyl", "severity": "Contraindicated",
         "mechanism": "Both are CNS depressants. Combined use causes synergistic respiratory depression via GABA-A potentiation (clonazepam) and mu-opioid agonism (fentanyl). FDA black box warning.",
         "clinical_significance": "High risk of fatal respiratory depression, profound sedation, coma, and death. More than 30% of opioid overdose deaths involve benzodiazepines.",
         "management": "Avoid concurrent use. If co-prescribing is unavoidable, limit doses, duration, and monitor closely. Naloxone should be available.",
         "alternatives": "Non-benzodiazepine anxiolytics (buspirone, SSRIs) or non-opioid analgesics (NSAIDs, acetaminophen, nerve blocks)"},
        {"drug_a": "Clonazepam", "drug_b": "Oxycodone", "severity": "Contraindicated",
         "mechanism": "Synergistic CNS and respiratory depression. FDA black box warning for concurrent benzodiazepine and opioid use.",
         "clinical_significance": "Fatal respiratory depression risk. Leading cause of prescription drug overdose deaths.",
         "management": "Avoid. If necessary, use lowest effective doses with close monitoring and naloxone availability.",
         "alternatives": "Non-opioid pain management; non-benzodiazepine anxiolytics"},
        {"drug_a": "Clonazepam", "drug_b": "Morphine", "severity": "Contraindicated",
         "mechanism": "Synergistic CNS depression via GABA-A and mu-opioid pathways. FDA black box warning.",
         "clinical_significance": "Severe respiratory depression and death risk.",
         "management": "Avoid concurrent use.",
         "alternatives": "Non-opioid analgesics; non-benzodiazepine anxiolytics"},
        {"drug_a": "Alprazolam", "drug_b": "Fentanyl", "severity": "Contraindicated",
         "mechanism": "Synergistic CNS and respiratory depression. FDA black box warning.",
         "clinical_significance": "Fatal respiratory depression risk.",
         "management": "Avoid. Monitor closely if unavoidable.",
         "alternatives": "Non-benzodiazepine anxiolytics; non-opioid analgesics"},
        {"drug_a": "Diazepam", "drug_b": "Fentanyl", "severity": "Contraindicated",
         "mechanism": "Synergistic CNS and respiratory depression. FDA black box warning.",
         "clinical_significance": "Fatal respiratory depression risk.",
         "management": "Avoid. Monitor closely if unavoidable.",
         "alternatives": "Non-benzodiazepine anxiolytics; non-opioid analgesics"},
    ]


# ── 3. Build text chunks ────────────────────────────────────────────────────

def make_chunk(rec: dict) -> str:
    parts = [
        f"Drug interaction: {rec['drug_a']} + {rec['drug_b']}.",
        f"Severity: {rec['severity']}.",
        f"Mechanism: {rec['mechanism']}." if rec.get("mechanism") else "",
        f"Clinical significance: {rec['clinical_significance']}." if rec.get("clinical_significance") else "",
        f"Management: {rec['management']}." if rec.get("management") else "",
        f"Alternatives: {rec['alternatives']}" if rec.get("alternatives") else "",
    ]
    return " ".join(p for p in parts if p)


# ── 4. Embed via Ollama nomic-embed-text ─────────────────────────────────────

def embed_text(text: str) -> list[float]:
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts sequentially (Ollama doesn't support batch embedding)."""
    embeddings = []
    for text in texts:
        embeddings.append(embed_text(text))
    return embeddings


# ── 5-7. Main: build ChromaDB ───────────────────────────────────────────────

def main():
    # Try DDInter download, fall back to seed
    try:
        records = download_ddinter()
    except Exception as e:
        print(f"DDInter download failed: {e}")
        records = []

    if not records:
        records = build_seed_dataset()

    total = len(records)
    est_time_s = total * 0.05
    est_time_m = est_time_s / 60
    est_db_mb = total * 4 / 1024

    print(f"\n{'=' * 60}")
    print(f"  Records to embed: {total:,}")
    print(f"  Estimated time:   {est_time_m:.0f} minutes ({est_time_s:.0f}s at ~50ms/embed)")
    print(f"  Estimated DB size: {est_db_mb:.0f} MB (~4KB/record)")
    print(f"  Batch size:       {BATCH_SIZE}")
    print(f"{'=' * 60}\n")

    # Ensure nomic-embed-text is available
    print(f"Checking {EMBED_MODEL} model ...")
    try:
        test_resp = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": "test"},
            timeout=60,
        )
        test_resp.raise_for_status()
        print(f"  {EMBED_MODEL} ready (embedding dim: {len(test_resp.json()['embedding'])})")
    except Exception as e:
        print(f"ERROR: {EMBED_MODEL} not available. Run: ollama pull {EMBED_MODEL}")
        print(f"  Detail: {e}")
        sys.exit(1)

    # Init ChromaDB
    DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Delete existing collection if present
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing '{COLLECTION_NAME}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    t0 = time.time()
    embedded = 0
    failed = 0

    # Process in batches
    for batch_start in range(0, total, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total)
        batch = records[batch_start:batch_end]

        # Build chunks for this batch
        chunks = [make_chunk(rec) for rec in batch]

        # Embed batch
        embeddings = []
        for j, chunk in enumerate(chunks):
            try:
                embeddings.append(embed_text(chunk))
            except Exception as e:
                idx = batch_start + j
                print(f"  Embed failed for record {idx} ({batch[j]['drug_a']}+{batch[j]['drug_b']}): {e}")
                embeddings.append(None)
                failed += 1

        # Add successful embeddings to ChromaDB
        ids = []
        docs = []
        embs = []
        metas = []
        for j, (rec, emb) in enumerate(zip(batch, embeddings)):
            if emb is None:
                continue
            idx = batch_start + j
            ids.append(f"ddi_{idx:06d}")
            docs.append(chunks[j])
            embs.append(emb)
            metas.append({
                "drug_a": rec["drug_a"],
                "drug_b": rec["drug_b"],
                "severity": rec["severity"],
                "mechanism": rec.get("mechanism", ""),
                "management": rec.get("management", ""),
                "alternatives": rec.get("alternatives", ""),
            })

        if ids:
            collection.add(ids=ids, embeddings=embs, documents=docs, metadatas=metas)
            embedded += len(ids)

        # Progress every 1000 records
        if batch_end % 1000 == 0 or batch_end == total:
            elapsed = time.time() - t0
            rate = embedded / elapsed if elapsed > 0 else 0
            eta = (total - batch_end) / rate if rate > 0 else 0
            print(f"  Progress: {batch_end:,}/{total:,} embedded={embedded:,} failed={failed} "
                  f"({elapsed:.0f}s, {rate:.0f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0

    # DB size on disk
    db_size = sum(f.stat().st_size for f in DB_DIR.rglob("*") if f.is_file())
    db_size_mb = db_size / 1024 / 1024

    print(f"\n{'=' * 60}")
    print(f"  DDI ChromaDB Build Complete")
    print(f"  Total records embedded: {embedded:,}")
    print(f"  Failed: {failed}")
    print(f"  Time taken: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  DB size on disk: {db_size_mb:.1f} MB")
    print(f"  Location: {DB_DIR}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
