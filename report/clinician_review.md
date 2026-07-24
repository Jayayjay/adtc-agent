# IMCI extended + young-infant branches — clinician review packet

**Status: UNREVIEWED.** These classifications and dose bands drive training data ONLY with `scripts/generate_sft_data.py --include-extended`, which is OFF by default. Nothing here ships until a qualified clinician signs it off against the current national IMCI adaptation. This packet is generated from the code (the classifiers were run to produce every row) so it cannot drift from what the model was trained to say.

**Sources:** classifications transcribed in `data/imci_2022/classifications.json` (2022 SA adaptation, cross-checked vs the WHO 2014 generic); doses in `data/imci_2022/dosing_tables.json`. Severity uses the IMCI colour tiers.

**How to review each row:** confirm the *trigger* and the *severity* against the national chart; note any correction in the last column; tick the section sign-off box. For doses, confirm each band boundary and value.


## Part A / B — classifications (severity + trigger + action, as implemented)


### Fever / malaria

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `very_severe_febrile_disease` | PINK (refer urgently) | Fever with a general danger sign or a stiff neck classifies as very severe febrile disease, whatever the malaria test shows. | Give the first dose of an appropriate antibiotic. Treat the child to prevent low blood sugar. Give paracetamol for high fever. Refer URGENTLY to hospital. | ceftriaxone_im, paracetamol | |
| `malaria` | YELLOW (treat + follow up) | The malaria test is positive and there are no danger signs or stiff neck. | Give a first-line oral antimalarial as the national guideline directs. Give paracetamol for high fever. Advise the mother when to return immediately. Follow up in 3 days if the fever persists. | artemether_lumefantrine, paracetamol | |
| `fever_no_malaria` | GREEN (home care) | The malaria test is negative, so this fever is not malaria. | Look for and treat another cause of fever. Give paracetamol for high fever. Advise the mother when to return immediately. Follow up in 3 days if the fever persists. | paracetamol | |
| `fever_malaria_test_required` | GREEN (home care) | This child has fever in a malaria risk area, or has travelled to one, and no malaria test has been done. The chart booklet requires a malaria test before the fever can be classified. | Do a malaria test (RDT or microscopy) now and classify on the result. Give paracetamol for high fever meanwhile, and refer urgently if any danger sign appears. | paracetamol | |

- [ ] **Fever / malaria** reviewed and correct  ·  corrections: ______________________


### Measles

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `severe_complicated_measles` | PINK (refer urgently) | A generalised rash with cough, runny nose, or red eyes meets the measles case definition. There is clouding of the cornea, which makes this severe complicated measles. | Give vitamin A. Give the first dose of an appropriate antibiotic. If there is clouding of the cornea or pus draining from the eye, apply tetracycline eye ointment. Refer URGENTLY to hospital. | vitamin_a, amoxicillin | |
| `measles_with_eye_or_mouth_complications` | YELLOW (treat + follow up) | A generalised rash with cough, runny nose, or red eyes meets the measles case definition. There is pus draining from the eye. | Give vitamin A. If there is pus draining from the eye, apply tetracycline eye ointment. If there are mouth ulcers, treat with gentian violet. Follow up in 3 days. | vitamin_a | |
| `measles` | GREEN (home care) | A generalised rash with cough, runny nose, or red eyes meets the measles case definition. There are no eye or mouth complications and no danger signs. | Give vitamin A. Advise the mother when to return immediately. | vitamin_a | |

- [ ] **Measles** reviewed and correct  ·  corrections: ______________________


### Anaemia

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `severe_anaemia` | PINK (refer urgently) | There is severe palmar pallor. | Refer URGENTLY to hospital. | — | |
| `anaemia` | YELLOW (treat + follow up) | There is some palmar pallor. | Give iron. Give mebendazole if the child is 1 year or older and has not had a dose in the last 6 months. Advise the mother when to return immediately. Follow up in 14 days. | iron | |
| `no_anaemia` | GREEN (home care) | There is no palmar pallor. | No treatment for anaemia is needed. | — | |

- [ ] **Anaemia** reviewed and correct  ·  corrections: ______________________


### Acute malnutrition

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `complicated_severe_acute_malnutrition` | PINK (refer urgently) | There is oedema of both feet, which is always severe acute malnutrition. With oedema of both feet, this is complicated severe acute malnutrition. | Give the first dose of an appropriate antibiotic. Treat the child to prevent low blood sugar. Keep the child warm. Refer URGENTLY to hospital. | vitamin_a | |
| `uncomplicated_severe_acute_malnutrition` | YELLOW (treat + follow up) | MUAC is 110mm, below 115mm. The appetite test passed and there is no medical complication. | Give ready-to-use therapeutic food (RUTF) for the child to take at home. Assess and counsel on feeding. Advise the mother when to return immediately. Follow up in 7 days. | amoxicillin, vitamin_a | |
| `moderate_acute_malnutrition` | YELLOW (treat + follow up) | MUAC is 120mm, between 115mm and 125mm. | Assess and counsel on feeding. Advise the mother when to return immediately. Follow up in 30 days. | vitamin_a | |
| `no_acute_malnutrition` | GREEN (home care) | There is no oedema of both feet and no wasting by MUAC or weight-for-height. | If the child is less than 2 years old, assess and counsel on feeding. No treatment for acute malnutrition is needed. | — | |

- [ ] **Acute malnutrition** reviewed and correct  ·  corrections: ______________________


### Wheeze (cough sub-branch)

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `wheeze_with_danger_sign` | PINK (refer urgently) | The child has wheeze. There is a general danger sign, so give salbutamol and refer urgently. | Give salbutamol by spacer. Give the first dose of an appropriate antibiotic and refer URGENTLY to hospital. | ceftriaxone_im | |
| `wheeze` | YELLOW (treat + follow up) | The child has wheeze. | Give salbutamol by spacer for 5 days. Follow up in 5 days if the child is still wheezing. If a cough lasts more than 14 days or the wheeze is recurrent, assess for TB or asthma. | — | |

- [ ] **Wheeze (cough sub-branch)** reviewed and correct  ·  corrections: ______________________


### Persistent diarrhoea (>=14 days)

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `severe_persistent_diarrhoea` | PINK (refer urgently) | Diarrhoea has lasted 14 days or more and dehydration is present. | Treat dehydration before referral unless the child has another severe classification. Give an extra dose of vitamin A. Refer URGENTLY to hospital. | vitamin_a | |
| `persistent_diarrhoea` | YELLOW (treat + follow up) | Diarrhoea has lasted 14 days or more with no visible dehydration. | Counsel the caregiver on feeding for persistent diarrhoea. Give an extra dose of vitamin A and give zinc for 14 days. Follow up in 5 days. | zinc, vitamin_a | |

- [ ] **Persistent diarrhoea (>=14 days)** reviewed and correct  ·  corrections: ______________________


### Dysentery (blood in stool)

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `severe_dysentery` | PINK (refer urgently) | There is blood in the stool and the child is less than 12 months old. | Treat the child to prevent low blood sugar, keep the child warm, and refer URGENTLY to hospital. | — | |
| `dysentery` | YELLOW (treat + follow up) | There is blood in the stool, the child is 12 months or older, and there is no dehydration. | Treat for 3 days with ciprofloxacin. Advise the mother when to return immediately. Follow up in 2 days. | — | |

- [ ] **Dysentery (blood in stool)** reviewed and correct  ·  corrections: ______________________


### Sore throat (from 3 years)

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `streptococcal_sore_throat` | YELLOW (treat + follow up) | There is enlarged tonsils, white or yellow exudate on the tonsils, with no runny nose and no cough, which points to a streptococcal sore throat. | Give penicillin. Treat pain and fever. Soothe the throat with a safe remedy. Follow up in 5 days if symptoms are worse or not resolving. | — | |
| `sore_throat_non_streptococcal` | GREEN (home care) | There are not enough signs to classify this as a streptococcal sore throat. | Soothe the throat with a safe remedy. | — | |

- [ ] **Sore throat (from 3 years)** reviewed and correct  ·  corrections: ______________________


### Growth problem (RTHB curve)

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `growth_problem` | YELLOW (treat + follow up) | On the weight curve, the child is losing weight. | Assess feeding and counsel the caregiver on the feeding recommendations. Deworm and give vitamin A if due. Advise the mother when to return immediately. Follow up in 7 days if there is a feeding problem, otherwise in 14 days. | — | |

- [ ] **Growth problem (RTHB curve)** reviewed and correct  ·  corrections: ______________________


### HIV

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `confirmed_hiv_infection` | YELLOW (treat + follow up) | The child has a positive HIV test or is already on ART, so HIV infection is confirmed. | Follow the steps to initiate ART. Give cotrimoxazole prophylaxis from 6 weeks of age. Ask about the caregiver's health and manage appropriately. Provide long-term follow-up. ART initiation is not urgent -- stabilise any severe illness first. | — | |
| `hiv_exposed` | YELLOW (treat + follow up) | The child is HIV-exposed: on ARV prophylaxis, or with a negative test while still breastfeeding or within 6 weeks of breastfeeding, so infection is not yet ruled out. | Complete the appropriate infant ARV prophylaxis. Repeat HIV PCR testing per the schedule and reclassify on the result. Ask about the caregiver's health and provide follow-up care. | — | |
| `suspected_symptomatic_hiv` | YELLOW (treat + follow up) | There are 3 features of HIV infection (three or more), so symptomatic HIV is suspected. | Counsel and offer HIV testing for the child and reclassify on the result. Counsel the caregiver about her own health and offer testing. Provide long-term follow-up. | — | |
| `possible_hiv_infection` | YELLOW (treat + follow up) | HIV infection is possible because the mother is HIV-positive. | Provide routine care including HIV testing for the child. Counsel the caregiver about her health and offer testing and treatment as needed. Reclassify on the test result. | — | |
| `hiv_infection_unlikely` | GREEN (home care) | The HIV test is negative, all breastfeeding stopped 6 weeks or more before the test, and there are no features of HIV infection. | Provide routine care. Repeat HIV testing only if new features appear or exposure is ongoing. | — | |

- [ ] **HIV** reviewed and correct  ·  corrections: ______________________


### Young infant: bacterial infection

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `yi_very_severe_disease` | PINK (refer urgently) | The young infant has a bulging fontanelle, which is very severe disease. | Give the first dose of an appropriate intramuscular antibiotic. Treat to prevent low blood sugar. Keep the infant warm on the way to hospital. Refer URGENTLY. | — | |
| `yi_local_bacterial_infection` | YELLOW (treat + follow up) | The young infant has a red umbilicus, a local bacterial infection. | Give an appropriate oral antibiotic. If there is eye discharge, give an eye ointment. Teach the caregiver to treat the local infection at home and to give home care. Follow up in 2 days. | — | |
| `yi_no_bacterial_infection` | GREEN (home care) | No signs of very severe disease or a local bacterial infection. | Counsel the caregiver on home care for the young infant. | — | |

- [ ] **Young infant: bacterial infection** reviewed and correct  ·  corrections: ______________________


### Young infant: jaundice

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `yi_severe_jaundice` | PINK (refer urgently) | There is jaundice under 24 hours of age, which is severe jaundice. | Treat to prevent low blood sugar. Keep the infant warm. Refer URGENTLY to hospital. | — | |
| `yi_jaundice` | YELLOW (treat + follow up) | There is jaundice appearing after 24 hours of age, with palms and soles not yellow. | Advise the caregiver to return immediately if the palms and soles become yellow. Follow up in 1 day. If the infant is older than 14 days, refer for assessment. | — | |

- [ ] **Young infant: jaundice** reviewed and correct  ·  corrections: ______________________


### Young infant: diarrhoea

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `yi_severe_dehydration` | PINK (refer urgently) | There is diarrhoea with the infant is less than 1 month old, which is severe dehydration in a young infant. | Start intravenous fluids (Plan C). Give the first dose of an intramuscular antibiotic. Keep the infant warm on the way to hospital. Refer URGENTLY. | — | |
| `yi_dysentery` | PINK (refer urgently) | There is blood in the stool of a young infant. | Keep the infant warm on the way to hospital. Refer URGENTLY. | — | |
| `yi_severe_persistent_diarrhoea` | PINK (refer urgently) | Diarrhoea has lasted 14 days or more in a young infant. | Treat any dehydration before referral. Keep the infant warm. Refer URGENTLY. | — | |
| `yi_some_dehydration` | YELLOW (treat + follow up) | There is diarrhoea with two of restless/irritable, sunken eyes, and a slow skin pinch, which is some dehydration. | Give fluid for some dehydration (Plan B). Advise the mother to continue breastfeeding. Give zinc for 14 days. Follow up in 2 days. | — | |
| `yi_no_dehydration` | GREEN (home care) | There is diarrhoea with not enough signs for some or severe dehydration. | Give fluid and continue breastfeeding at home (Plan A). Give zinc for 14 days. Counsel the caregiver on home care. Follow up in 2 days. | — | |

- [ ] **Young infant: diarrhoea** reviewed and correct  ·  corrections: ______________________


### Young infant: congenital problems

| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |
|---|---|---|---|---|---|
| `yi_congenital_priority` | PINK (refer urgently) | There is a cleft lip or palate, a congenital priority sign. | Give any needed pre-referral treatment. Treat to prevent low blood sugar. Keep the infant warm. Refer URGENTLY to hospital. | — | |
| `yi_congenital_abnormal_signs` | YELLOW (treat + follow up) | There is a club foot, an abnormal sign. | Keep the infant warm, skin to skin. Assess breastfeeding and support the mother. Refer for assessment. | — | |
| `yi_possible_congenital_syphilis` | YELLOW (treat + follow up) | The mother's RPR is positive and she was untreated or only partially treated, so congenital syphilis is possible. | Check for signs of congenital syphilis and refer to hospital if present. If there are no signs, give intramuscular penicillin. Ensure the mother receives full treatment. | — | |

- [ ] **Young infant: congenital problems** reviewed and correct  ·  corrections: ______________________


## Part C — dosing tables (every band)

Doses are NOT graded by the model scorer; they are emitted verbatim from these tables. Confirm every band.


### amoxicillin — pneumonia, acute ear infection  
route: oral  ·  two times daily for 5 days  ·  source: 2014 p16  ·  keyed by weight

| weight (kg) | tablet_250mg | syrup_250mg_per_5ml_ml | OK? |
|---|---|---|---|
| 4–10 | 1 | 5 | |
| 10–14 | 2 | 10 | |
| 14–19 | 3 | 15 | |

### cotrimoxazole_prophylaxis — prophylaxis in HIV confirmed or exposed child  
route: oral  ·  once a day, starting at 4-6 weeks of age  ·  source: 2014 p16  ·  keyed by age

| age (months) | syrup_40_200_per_5ml_ml | paed_tablet_20_100 | adult_tablet_80_400 | OK? |
|---|---|---|---|---|
| 1.5–6 | 2.5 | 0.5 | 0.25 | |
| 6–60 | 5 | 2 | 0.5 | |

### paracetamol — high fever (>38.5C) or ear pain  
route: oral  ·  every 6 hours until fever or pain is gone  ·  source: 2014 p16  ·  keyed by weight

| weight (kg) | tablet_100mg | tablet_500mg | OK? |
|---|---|---|---|
| 4–14 | 1 | 0.25 | |
| 14–19 | 1.5 | 0.5 | |

### zinc — diarrhoea (age 2 months up to 5 years)  
route: oral  ·  daily for 14 days  ·  source: 2014 (GIVE EXTRA FLUID FOR DIARRHOEA)  ·  keyed by age

| age (months) | tablets_daily | OK? |
|---|---|---|
| 2–6 | 0.5 | |
| 6–∞ | 1 | |

### ors_plan_a_extra_fluid — diarrhoea, no dehydration -- amount after each loose stool  
route: oral  ·    ·  source: 2014 Plan A  ·  keyed by age

| age (months) | ml_after_each_loose_stool_min | ml_after_each_loose_stool_max | OK? |
|---|---|---|---|
| 0–24 | 50 | 100 | |
| 24–∞ | 100 | 200 | |

### vitamin_a — TREATMENT dose for measles / persistent diarrhoea (also the supplementation dose for >=6 months). NOT if dosed in the past month or on RUTF.  
route: oral  ·    ·  source: 2014 (Vitamin A supplementation and treatment); <6mo treatment band added per clinician 2026-07-22  ·  keyed by age

| age (months) | iu | OK? |
|---|---|---|
| 0–6 | 50000 | |
| 6–12 | 100000 | |
| 12–∞ | 200000 | |

### mebendazole — deworming if hookworm/whipworm endemic, child >=1 year, no dose in previous 6 months  
route: oral  ·  single dose in clinic  ·  source: 2014  ·  keyed by age

| age (months) | dose_mg | OK? |
|---|---|---|
| 12–∞ | 500 | |

### iron — anaemia (some palmar pallor, or Hb 7-11 g/dl)  
route: oral  ·  once daily for 3 months  ·  source: clinician 2026-07-22 (WHO IMCI Give Iron)  ·  keyed by flat

| applies to | dose_text | OK? |
|---|---|---|
| all | ferrous sulfate, 3 mg/kg/day of elemental iron | |

### ceftriaxone_im — sick-child pre-referral (very severe disease, suspected meningitis, severe pneumonia, mastoiditis)  
route: IM  ·    ·  source: 2022 p35  ·  keyed by weight
  
note: >17.5 kg: give 2 ml in each thigh; for weights over 17.5 kg dilute 1 g in 3.5 ml sterile water and give 5.5 ml IM
  
dilution: dilute 250 mg vial with 1 ml sterile water, or 500 mg with 2 ml sterile water (250 mg/ml)

| weight (kg) | dose_mg | volume_ml | OK? |
|---|---|---|---|
| 3.5–5.5 | 312 | 1.25 | |
| 5.5–7 | 440 | 1.75 | |
| 7–9 | 625 | 2.5 | |
| 9–11 | 750 | 3 | |
| 11–14 | 810 | 3.25 | |
| 14–17.5 | 1000 | 4 | |
| 17.5–∞ | 1500 | 5.5 | |

### diazepam_rectal — convulsing now  
route: rectal  ·    ·  source: 2022 p35  ·  keyed by weight
  
note: 0.5 mg/kg per rectum; repeat after 10 minutes if not stopped

| weight (kg) | dose_mg | volume_ml | OK? |
|---|---|---|---|
| 3–4 | 2 | 0.4 | |
| 4–5 | 2.5 | 0.5 | |
| 5–15 | 5 | 1 | |
| 15–25 | 7.5 | 1.5 | |

### young_infant_ampicillin_im — young infant (0-2 months) very severe disease -- with gentamicin  
route: IM  ·    ·  source: 2014 p51  ·  keyed by weight
  
note: 2014 young-infant regimen. Dose 50 mg/kg. 250 mg vial + 1.3 ml sterile water = 250 mg/1.5 ml.

| weight (kg) | volume_ml | OK? |
|---|---|---|
| 1–1.5 | 0.4 | |
| 1.5–2 | 0.5 | |
| 2–2.5 | 0.7 | |
| 2.5–3 | 0.8 | |
| 3–3.5 | 1.0 | |
| 3.5–4 | 1.1 | |
| 4–4.5 | 1.3 | |

### artemether_lumefantrine — uncomplicated malaria (malaria test positive)  
route: oral  ·  two times daily for 3 days  ·  source: 2014 p16  ·  keyed by weight
  
note: first dose in clinic, observe 1 hour, repeat if vomits within an hour; take with food

| weight (kg) | tablets_per_dose | OK? |
|---|---|---|
| 5–10 | 1 | |
| 10–14 | 1 | |
| 14–19 | 2 | |

- [ ] **All dosing tables** reviewed and correct  ·  corrections: ______________


## Part D — specific things to check (from the code review checklists)

- [ ] Confirm every weight/age band boundary against the source table (off-by-one on a band edge is a dosing error).
- [ ] Ceftriaxone dilution and per-thigh split (>17.5kg) -- 2022 p35.
- [ ] 2014 uses ampicillin+gentamicin for the young infant; 2022 sick-child pre-referral is ceftriaxone. Confirm which the national adaptation wants.
- [ ] ART regimens are edition- and country-specific -- verify against current national ART guidelines, not this booklet.
- [ ] Zinc, ORS volumes, vitamin A, mebendazole are stable across editions but confirm.
- [ ] Fever severe label kept as `very_severe_febrile_disease` (2014) + bulging fontanelle added — acceptable?
- [ ] HIV severity: all tiers MODERATE except `hiv_infection_unlikely` (MILD), none PINK (ART not urgent) — acceptable?
- [ ] Young-infant treatment emits NO specific dose (IM antibiotics / referral only) — acceptable?

See `src/sft/extended_protocol.py` and `src/sft/young_infant.py` docstrings for the full per-branch checklist items (numbered 1–12 and 1–5 respectively).


---
Reviewer: ____________________  Signature: ____________________  Date: __________
