# NHANES Dataset Download Instructions

This project uses data from the **National Health and Nutrition Examination Survey (NHANES)** conducted by the **Centers for Disease Control and Prevention (CDC)**.

NHANES data is publicly available but is **not included** in this repository due to licensing and size constraints.

---

## 1. NHANES Cycle Used

This study uses data from the **NHANES 2017–2018 cycle**.

All NHANES files are provided in **SAS Transport (.XPT) format**.

---

## 2. Required NHANES Files

Download the following files from the official NHANES website:

| File Name | Component | Purpose |
|----------|----------|---------|
| `DEMO_L.XPT` | Demographics | Age, Gender |
| `BMX_L.XPT` | Body Measures | BMI |
| `BPQ_L.XPT` | Blood Pressure Questionnaire | Hypertension |
| `DIQ_L.XPT` | Diabetes Questionnaire | Diabetes label |
| `SMQ_L.XPT` | Smoking Questionnaire | Smoking history |

---

## 3. Download Steps

1. Visit the official NHANES website:  
   https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx

2. Select the **2017–2018** survey cycle.

3. Download the required `.XPT` files listed above.


> File names must match exactly. Do not rename the files.


## 4. Loading NHANES Data in Python

NHANES files can be loaded directly using `pandas`:

```python
import pandas as pd

demo = pd.read_sas("data/nhanes/DEMO_L.XPT")
bmx  = pd.read_sas("data/nhanes/BMX_L.XPT")
bpx  = pd.read_sas("data/nhanes/BPQ_L.XPT")
diq  = pd.read_sas("data/nhanes/DIQ_L.XPT")
smq  = pd.read_sas("data/nhanes/SMQ_L.XPT")

## 5. Data Cleaning and Label Construction

Only participants with complete records across all selected components are retained.

Diabetes label is derived from the questionnaire variable DIQ010:

1 → Diabetic

2 → Non-diabetic

Hypertension is derived from BPQ020.

Smoking history is constructed using SMQ020 and SMQ040.

Records with missing or ambiguous responses are removed.

## 6. Feature Selection Notes

Laboratory diagnostic biomarkers such as HbA1c and blood glucose are intentionally excluded.

The study focuses on early-stage diabetes risk prediction using non-invasive and lifestyle-based features.

## 7. Important Notes

Do not manually convert .XPT files to CSV.

NHANES data is de-identified and publicly accessible.

Redistribution of NHANES data is not permitted.

## 8. NHANES Citation

If you use NHANES data, please cite:

Centers for Disease Control and Prevention (CDC).
National Health and Nutrition Examination Survey.
https://www.cdc.gov/nchs/nhanes/
4. Place all downloaded files in the following directory structure:

