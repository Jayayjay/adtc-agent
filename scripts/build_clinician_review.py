"""
Builds report/clinician_review.md -- a review packet for a qualified clinician
to sign off (or correct) the UNREVIEWED extended + young-infant branches and the
dosing tables before they train a shipped model.

Generated from the source of truth, not hand-written: every classification's
severity, trigger reasoning, and action are extracted by RUNNING the
deterministic classifier on the same structured cases the extended eval uses
(eval/tasks/imci_vignettes_extended.json), and the dose bands come straight from
data/imci_2022/dosing_tables.json. So the packet can never drift from what the
code actually does.

Usage:
    python scripts/build_clinician_review.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sft.extended_protocol import (
    ExtendedAssessment,
    classify_anaemia, classify_dysentery, classify_fever_malaria, classify_growth,
    classify_hiv, classify_malnutrition, classify_measles, classify_persistent_diarrhoea,
    classify_sore_throat, classify_wheeze,
)
from src.sft.treatment import LABEL_TO_DRUGS
from src.sft.young_infant import (
    YoungInfantAssessment,
    classify_yi_bacterial, classify_yi_congenital, classify_yi_diarrhoea, classify_yi_jaundice,
)
from src.tools import imci_dosing

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "report" / "clinician_review.md"
OUT_HTML = REPO / "report" / "clinician_review.html"
OUT_TEX = REPO / "report" / "clinician_review.tex"

CLASSIFIERS = {
    "classify_wheeze": classify_wheeze, "classify_persistent_diarrhoea": classify_persistent_diarrhoea,
    "classify_dysentery": classify_dysentery, "classify_sore_throat": classify_sore_throat,
    "classify_growth": classify_growth, "classify_fever_malaria": classify_fever_malaria,
    "classify_measles": classify_measles, "classify_anaemia": classify_anaemia,
    "classify_malnutrition": classify_malnutrition, "classify_hiv": classify_hiv,
    "classify_yi_bacterial": classify_yi_bacterial, "classify_yi_jaundice": classify_yi_jaundice,
    "classify_yi_diarrhoea": classify_yi_diarrhoea, "classify_yi_congenital": classify_yi_congenital,
}

# branch grouping for the packet: label -> (section, human title)
SECTIONS = [
    ("Fever / malaria", ExtendedAssessment, classify_fever_malaria,
     [dict(age_months=24, fever=True, stiff_neck=True),
      dict(age_months=36, fever=True, malaria_risk="high", malaria_test="positive"),
      dict(age_months=36, fever=True, malaria_test="negative"),
      dict(age_months=36, fever=True, malaria_risk="high", malaria_test="not_done", travel_to_malaria_area=True)]),
    ("Measles", ExtendedAssessment, classify_measles,
     [dict(age_months=24, generalised_rash=True, cough_or_runny_nose_or_red_eyes=True, clouding_of_cornea=True),
      dict(age_months=24, generalised_rash=True, cough_or_runny_nose_or_red_eyes=True, pus_draining_from_eye=True),
      dict(age_months=24, generalised_rash=True, cough_or_runny_nose_or_red_eyes=True)]),
    ("Anaemia", ExtendedAssessment, classify_anaemia,
     [dict(age_months=24, severe_palmar_pallor=True), dict(age_months=24, some_palmar_pallor=True),
      dict(age_months=24)]),
    ("Acute malnutrition", ExtendedAssessment, classify_malnutrition,
     [dict(age_months=24, oedema_of_both_feet=True),
      dict(age_months=24, muac_mm=110, appetite_test_passed=True),
      dict(age_months=24, muac_mm=120), dict(age_months=24, muac_mm=140)]),
    ("Wheeze (cough sub-branch)", ExtendedAssessment, classify_wheeze,
     [dict(age_months=24, wheeze=True, danger_signs_present=["lethargic_or_unconscious"]),
      dict(age_months=18, wheeze=True)]),
    ("Persistent diarrhoea (>=14 days)", ExtendedAssessment, classify_persistent_diarrhoea,
     [dict(age_months=30, diarrhoea=True, diarrhoea_days=20, dehydration_present=True),
      dict(age_months=30, diarrhoea=True, diarrhoea_days=18)]),
    ("Dysentery (blood in stool)", ExtendedAssessment, classify_dysentery,
     [dict(age_months=8, diarrhoea=True, blood_in_stool=True),
      dict(age_months=36, diarrhoea=True, blood_in_stool=True)]),
    ("Sore throat (from 3 years)", ExtendedAssessment, classify_sore_throat,
     [dict(age_months=48, sore_throat=True, enlarged_tonsils=True, tonsil_exudate=True),
      dict(age_months=48, sore_throat=True, runny_nose=True, cough=True)]),
    ("Growth problem (RTHB curve)", ExtendedAssessment, classify_growth,
     [dict(age_months=30, losing_weight=True)]),
    ("HIV", ExtendedAssessment, classify_hiv,
     [dict(age_months=24, hiv_test="positive"),
      dict(age_months=6, infant_on_arv_prophylaxis=True),
      dict(age_months=30, hiv_oral_thrush=True, hiv_low_weight=True, hiv_pneumonia_now=True),
      dict(age_months=24, mother_hiv_positive=True),
      dict(age_months=24, hiv_test="negative", breastfeeding_stopped_ge_6wk=True)]),
    ("Young infant: bacterial infection", YoungInfantAssessment, classify_yi_bacterial,
     [dict(age_days=20, bulging_fontanelle=True), dict(age_days=20, umbilicus_red_only=True),
      dict(age_days=20)]),
    ("Young infant: jaundice", YoungInfantAssessment, classify_yi_jaundice,
     [dict(age_days=0, jaundice=True, jaundice_onset_under_24h=True), dict(age_days=5, jaundice=True)]),
    ("Young infant: diarrhoea", YoungInfantAssessment, classify_yi_diarrhoea,
     [dict(age_days=20, diarrhoea=True), dict(age_days=30, diarrhoea=True, blood_in_stool=True),
      dict(age_days=40, diarrhoea=True, diarrhoea_days=16),
      dict(age_days=45, diarrhoea=True, restless_or_irritable=True, skin_pinch_slow=True),
      dict(age_days=45, diarrhoea=True)]),
    ("Young infant: congenital problems", YoungInfantAssessment, classify_yi_congenital,
     [dict(age_days=2, cleft_lip_or_palate=True), dict(age_days=2, club_foot=True),
      dict(age_days=2, mother_rpr_positive_untreated=True)]),
]

TIER = {"severe": "PINK (refer urgently)", "moderate": "YELLOW (treat + follow up)", "green": "GREEN",
        "mild": "GREEN (home care)"}


def _drugs_for(label: str) -> str:
    drugs = LABEL_TO_DRUGS.get(label, [])
    return ", ".join(drugs) if drugs else "—"


_SEV_TIER = {"severe": ("refer", "PINK · refer urgently"),
             "moderate": ("treat", "YELLOW · treat + follow up"),
             "mild": ("home", "GREEN · home care")}


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


_CSS = """
:root{
  --bg:#f7f9fb; --panel:#ffffff; --ink:#16232b; --muted:#5b6b74; --line:#dde5ea;
  --accent:#0e7490; --accent-ink:#0b5566;
  --refer:#c2255c; --refer-bg:#fdeef3; --treat:#b45309; --treat-bg:#fbf1e4; --home:#2f8f4e; --home-bg:#ecf7ef;
  --warn:#b4530a; --warn-bg:#fbeee1;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0f1417; --panel:#161d22; --ink:#e6edf1; --muted:#93a2ab; --line:#26313a;
  --accent:#3bb3c9; --accent-ink:#7fd3e2;
  --refer:#f06595; --refer-bg:#2a1620; --treat:#e0902f; --treat-bg:#271c10; --home:#5fc57e; --home-bg:#12241a;
  --warn:#e0902f; --warn-bg:#271c10;
}}
:root[data-theme="dark"]{
  --bg:#0f1417; --panel:#161d22; --ink:#e6edf1; --muted:#93a2ab; --line:#26313a;
  --accent:#3bb3c9; --accent-ink:#7fd3e2;
  --refer:#f06595; --refer-bg:#2a1620; --treat:#e0902f; --treat-bg:#271c10; --home:#5fc57e; --home-bg:#12241a;
  --warn:#e0902f; --warn-bg:#271c10;
}
:root[data-theme="light"]{
  --bg:#f7f9fb; --panel:#ffffff; --ink:#16232b; --muted:#5b6b74; --line:#dde5ea;
  --accent:#0e7490; --accent-ink:#0b5566;
  --refer:#c2255c; --refer-bg:#fdeef3; --treat:#b45309; --treat-bg:#fbf1e4; --home:#2f8f4e; --home-bg:#ecf7ef;
  --warn:#b4530a; --warn-bg:#fbeee1;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55;
  font-size:15px;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:40px 24px 80px}
header.top{border-bottom:2px solid var(--accent);padding-bottom:20px;margin-bottom:8px}
.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--accent-ink);font-weight:700}
h1{font-size:30px;line-height:1.15;margin:.3em 0 .2em;text-wrap:balance;letter-spacing:-.01em}
h2{font-size:20px;margin:44px 0 12px;padding-top:14px;border-top:1px solid var(--line);letter-spacing:-.01em}
h3{font-size:15px;margin:22px 0 6px;font-family:var(--mono);color:var(--accent-ink)}
p{color:var(--muted);max-width:68ch}
.banner{display:flex;gap:12px;align-items:flex-start;background:var(--warn-bg);border:1px solid var(--warn);
  border-radius:10px;padding:14px 16px;margin:18px 0;color:var(--ink)}
.banner b{color:var(--warn)}
.meta{font-size:13px;color:var(--muted);margin:6px 0}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;margin:10px 0 6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{text-align:left;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
  font-weight:700;padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap;background:var(--panel);position:sticky;top:0}
td{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:none}
td.num,th.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
code{font-family:var(--mono);font-size:12.5px;background:var(--refer-bg);background:color-mix(in srgb,var(--line) 45%,transparent);
  padding:1px 5px;border-radius:5px}
.pill{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.03em;padding:2px 9px;border-radius:999px;white-space:nowrap}
.pill.refer{color:var(--refer);background:var(--refer-bg);border:1px solid var(--refer)}
.pill.treat{color:var(--treat);background:var(--treat-bg);border:1px solid var(--treat)}
.pill.home{color:var(--home);background:var(--home-bg);border:1px solid var(--home)}
td.tier{border-left:4px solid var(--line)}
tr.refer td.tier{border-left-color:var(--refer)} tr.treat td.tier{border-left-color:var(--treat)} tr.home td.tier{border-left-color:var(--home)}
.chk{width:18px;height:18px;border:1.5px solid var(--muted);border-radius:4px;display:inline-block}
.corr{min-width:120px;color:var(--muted)}
ul.check{list-style:none;padding:0;margin:10px 0}
ul.check li{display:flex;gap:10px;align-items:flex-start;padding:7px 0;border-bottom:1px solid var(--line);max-width:80ch}
.signoff{margin-top:40px;border:1px solid var(--accent);border-radius:12px;padding:20px 22px;background:var(--panel)}
.signoff .row{display:flex;flex-wrap:wrap;gap:28px;margin-top:14px}
.signoff .field{flex:1;min-width:180px;border-bottom:1.5px solid var(--muted);padding-bottom:4px;font-size:12px;color:var(--muted)}
@media print{
  body{background:#fff;color:#000;font-size:11px}
  .wrap{max-width:none;padding:0}
  .tablewrap,.signoff,.banner{border-color:#999}
  h2{break-before:auto} tr{break-inside:avoid}
  th{position:static}
}
"""


def emit_html() -> str:
    P = ['<style>' + _CSS + '</style>', '<div class="wrap">']
    P.append('<header class="top">')
    P.append('<div class="eyebrow">IMCI · clinical sign-off</div>')
    P.append('<h1>Extended &amp; young-infant branches — clinician review</h1>')
    P.append('<div class="banner"><span>⚠️</span><div><b>UNREVIEWED.</b> These classifications and dose '
             'bands train a model only with <code>--include-extended</code> (off by default). Nothing '
             'ships until signed off against the current national IMCI adaptation. Every row here is '
             'generated from the code, so it matches exactly what the model was trained to say.</div></div>')
    P.append('<p class="meta">Sources: <code>data/imci_2022/classifications.json</code> (2022 SA '
             'adaptation, cross-checked vs the WHO 2014 generic) · doses in '
             '<code>data/imci_2022/dosing_tables.json</code>. Severity uses the IMCI colour tiers.</p>')
    P.append('<p class="meta">To review: confirm each <b>trigger</b> and <b>severity</b> against the '
             'national chart, tick <b>OK?</b> or write a correction, then sign the block at the end.</p>')
    P.append('</header>')

    P.append('<h2>Classifications — severity, trigger &amp; action (as implemented)</h2>')
    for title, cls, clf, cases in SECTIONS:
        P.append(f'<h3>{_esc(title)}</h3>')
        P.append('<div class="tablewrap"><table><thead><tr>'
                 '<th>Classification</th><th>Severity</th><th>Trigger (as coded)</th>'
                 '<th>Action</th><th>Drugs</th><th>OK?</th><th>Correction</th></tr></thead><tbody>')
        seen = set()
        for inp in cases:
            r = clf(cls(**inp))
            if r is None or r.condition_label in seen:
                continue
            seen.add(r.condition_label)
            klass, pill = _SEV_TIER.get(r.classification.value, ("treat", r.classification.value))
            P.append(f'<tr class="{klass}">'
                     f'<td class="tier"><code>{_esc(r.condition_label)}</code></td>'
                     f'<td><span class="pill {klass}">{pill}</span></td>'
                     f'<td>{_esc(" ".join(r.reasoning))}</td>'
                     f'<td>{_esc(r.recommended_action)}</td>'
                     f'<td>{_esc(_drugs_for(r.condition_label))}</td>'
                     f'<td><span class="chk"></span></td><td class="corr"></td></tr>')
        P.append('</tbody></table></div>')

    P.append('<h2>Dosing tables — confirm every band</h2>')
    P.append('<p class="meta">Doses are not graded by the model scorer; they are emitted verbatim from '
             'these tables.</p>')
    tables = imci_dosing._tables()["drugs"]
    for name, entry in tables.items():
        key = entry["key"]
        P.append(f'<h3>{_esc(name)}</h3>')
        sub = f"{entry.get('indication','')} · {entry.get('route','')} · {entry.get('frequency','')} · source {entry.get('source','')} · keyed by {key}"
        P.append(f'<p class="meta">{_esc(sub)}</p>')
        if entry.get("note"):
            P.append(f'<p class="meta">note: {_esc(entry["note"])}</p>')
        sample = entry["bands"][0]
        dose_fields = [k for k in sample if k not in
                       ("age_band", "weight_kg_min", "weight_kg_max", "age_months_min", "age_months_max")]
        rng = "weight (kg)" if key == "weight" else "age (months)"
        head = f'<th>{rng}</th>' + "".join(f'<th class="num">{_esc(f)}</th>' for f in dose_fields) + '<th>OK?</th>'
        P.append('<div class="tablewrap"><table><thead><tr>' + head + '</tr></thead><tbody>')
        for b in entry["bands"]:
            lo, hi = ((b.get("weight_kg_min"), b.get("weight_kg_max")) if key == "weight"
                      else (b.get("age_months_min"), b.get("age_months_max")))
            band = f"{lo}–{'&#8734;' if hi is None else hi}"
            cells = "".join(f'<td class="num">{_esc(str(b.get(f,"")))}</td>' for f in dose_fields)
            P.append(f'<tr><td class="num">{band}</td>{cells}<td><span class="chk"></span></td></tr>')
        P.append('</tbody></table></div>')

    P.append('<h2>Specific things to check</h2><ul class="check">')
    items = list(imci_dosing._tables()["_meta"]["review_checklist"]) + [
        "Fever severe label kept as very_severe_febrile_disease (2014) with bulging fontanelle added — acceptable?",
        "HIV severity: all tiers MODERATE except hiv_infection_unlikely (MILD), none PINK (ART not urgent) — acceptable?",
        "Young-infant treatment emits no specific dose (IM antibiotics / referral only) — acceptable?",
    ]
    for it in items:
        P.append(f'<li><span class="chk"></span><span>{_esc(it)}</span></li>')
    P.append('</ul>')
    P.append('<p class="meta">Full per-branch checklist items are in the '
             '<code>src/sft/extended_protocol.py</code> and <code>src/sft/young_infant.py</code> docstrings.</p>')

    P.append('<div class="signoff"><b>Sign-off</b>'
             '<div class="row"><div class="field">Reviewer name</div>'
             '<div class="field">Signature</div><div class="field">Date</div></div>'
             '<div class="row"><div class="field">National adaptation reviewed against</div></div></div>')
    P.append('</div>')
    return "\n".join(P)


_TEX_UNICODE = {"—": "--", "–": "--", "≥": r"$\geq$", "≤": r"$\leq$", "·": r"\textperiodcentered{}",
                "∞": r"$\infty$", "→": r"$\rightarrow$", "×": r"$\times$", "°": r"\textdegree{}",
                "’": "'", "“": "``", "”": "''"}
_TEX_SPECIAL = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _tex(s: str) -> str:
    s = str(s)
    for u, r in _TEX_UNICODE.items():
        s = s.replace(u, r)
    return "".join(_TEX_SPECIAL.get(c, c) for c in s)


def _tex_code(s: str) -> str:
    """Like _tex, but lets a long label/code line-break at each underscore so it
    wraps inside a narrow column instead of overflowing into the next one."""
    return _tex(s).replace(r"\_", r"\_\allowbreak ")


def _fname(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s)


_TEX_TIER = {"severe": (r"\refer", "PINK -- refer"), "moderate": (r"\treat", "YELLOW -- treat"),
             "mild": (r"\home", "GREEN -- home")}

_TEX_PREAMBLE = r"""% Clinician review packet -- GENERATED by scripts/build_clinician_review.py.
% Editable/fillable PDF: compile with pdflatex (the checkboxes and text fields are
% AcroForm fields via hyperref; open the PDF in any reader that supports forms).
\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[a4paper,margin=1.5cm,top=1.6cm,bottom=1.6cm]{geometry}
\usepackage{longtable}
\usepackage{array}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage{amssymb}
\usepackage{enumitem}
\usepackage[colorlinks=false,pdfborder={0 0 0}]{hyperref}
\usepackage{titlesec}

\definecolor{refer}{HTML}{C2255C}
\definecolor{treat}{HTML}{B45309}
\definecolor{home}{HTML}{2F8F4E}
\definecolor{accent}{HTML}{0E7490}
\definecolor{warnbg}{HTML}{FBEEE1}
\definecolor{rule}{HTML}{DDE5EA}
\newcommand{\refer}[1]{\textcolor{refer}{\textbf{#1}}}
\newcommand{\treat}[1]{\textcolor{treat}{\textbf{#1}}}
\newcommand{\home}[1]{\textcolor{home}{\textbf{#1}}}
\titleformat{\section}{\large\bfseries\color{accent}}{}{0em}{}
\titleformat{\subsection}{\normalsize\bfseries}{}{0em}{}
\setlength{\parindent}{0pt}
\setlength{\extrarowheight}{2pt}
\renewcommand{\arraystretch}{1.15}
\hypersetup{pdftitle={IMCI extended + young-infant branches -- clinician review}}

\begin{document}
\begin{Form}
"""


def emit_tex() -> str:
    fid = [0]

    def ok_box(tag):
        fid[0] += 1
        return rf"\CheckBox[name=ok_{_fname(tag)}_{fid[0]},width=1.6ex,height=1.6ex]{{}}"

    def text_field(tag, height="1.2cm"):
        fid[0] += 1
        return (rf"\TextField[name=tf_{_fname(tag)}_{fid[0]},width=\linewidth,height={height},"
                rf"multiline=true,backgroundcolor={{1 1 0.98}}]{{}}")

    T = [_TEX_PREAMBLE]
    T.append(r"{\Large\bfseries\color{accent} IMCI extended \& young-infant branches --- clinician review}\\[2pt]")
    T.append(r"{\footnotesize\color{accent} IMCI \textperiodcentered{} clinical sign-off}\\[8pt]")
    T.append(r"\colorbox{warnbg}{\parbox{\dimexpr\linewidth-2\fboxsep}{\textbf{\color{treat}UNREVIEWED.} "
             r"These classifications and dose bands train a model only with \texttt{-{}-include-extended} "
             r"(off by default). Nothing ships until signed off against the current national IMCI "
             r"adaptation. Every row here is generated from the code, so it matches exactly what the "
             r"model was trained to say.}}\\[8pt]")
    T.append(r"{\footnotesize Sources: \texttt{data/\allowbreak imci\_2022/\allowbreak classifications.json} "
             r"(2022 SA adaptation, cross-checked vs the WHO 2014 generic); doses in "
             r"\texttt{data/\allowbreak imci\_2022/\allowbreak dosing\_tables.json}. "
             r"Severity uses the IMCI colour tiers. To review: confirm each \textbf{trigger} and "
             r"\textbf{severity}, tick \textbf{OK} or write a correction, then sign at the end.}\\[6pt]")

    T.append(r"\section*{Classifications --- severity, trigger \& action (as implemented)}")
    for title, cls, clf, cases in SECTIONS:
        T.append(rf"\subsection*{{{_tex(title)}}}")
        T.append(r"\begin{longtable}{@{}"
                 r">{\ttfamily\footnotesize\raggedright\arraybackslash}p{2.7cm} "
                 r">{\footnotesize\raggedright\arraybackslash}p{1.7cm} "
                 r">{\footnotesize\raggedright\arraybackslash}p{4.2cm} "
                 r">{\footnotesize\raggedright\arraybackslash}p{4.2cm} "
                 r">{\footnotesize\raggedright\arraybackslash}p{1.9cm} "
                 r">{\centering\arraybackslash}p{0.7cm}@{}}")
        T.append(r"\rowcolor{rule!40}\normalfont\footnotesize\textbf{Classification} & "
                 r"\textbf{Severity} & \textbf{Trigger (as coded)} & "
                 r"\textbf{Action} & \textbf{Drugs} & \textbf{OK}\\")
        T.append(r"\endhead")
        seen = set()
        for inp in cases:
            r = clf(cls(**inp))
            if r is None or r.condition_label in seen:
                continue
            seen.add(r.condition_label)
            cmd, txt = _TEX_TIER.get(r.classification.value, (r"\treat", r.classification.value))
            T.append(rf"{_tex_code(r.condition_label)} & {cmd}{{{txt}}} & "
                     rf"{_tex(' '.join(r.reasoning))} & "
                     rf"{_tex(r.recommended_action)} & "
                     rf"{_tex_code(_drugs_for(r.condition_label))} & {ok_box(r.condition_label)}\\")
        T.append(r"\end{longtable}")
        T.append(r"\vspace{3pt}")
        T.append(rf"{{\footnotesize\textbf{{Corrections for {_tex(title)}:}}}}\par\vspace{{2pt}}")
        T.append(text_field("corr_" + title))
        T.append(r"\vspace{9pt}")

    T.append(r"\section*{Dosing tables --- confirm every band}")
    T.append(r"{\footnotesize Doses are not graded by the model scorer; they are emitted verbatim from "
             r"these tables.}\\[4pt]")
    tables = imci_dosing._tables()["drugs"]
    for name, entry in tables.items():
        key = entry["key"]
        T.append(rf"\subsection*{{{_tex(name)}}}")
        T.append(rf"{{\footnotesize {_tex(entry.get('indication',''))} \textperiodcentered{{}} "
                 rf"{_tex(entry.get('route',''))} \textperiodcentered{{}} {_tex(entry.get('frequency',''))} "
                 rf"\textperiodcentered{{}} source {_tex(entry.get('source',''))} "
                 rf"\textperiodcentered{{}} keyed by {key}}}\\[-2pt]")
        if entry.get("note"):
            T.append(rf"{{\footnotesize note: {_tex(entry['note'])}}}\\[-2pt]")
        sample = entry["bands"][0]
        dose_fields = [k for k in sample if k not in
                       ("age_band", "weight_kg_min", "weight_kg_max", "age_months_min", "age_months_max")]
        rng = "weight (kg)" if key == "weight" else "age (months)"
        cols = "l " + " ".join(["r"] * len(dose_fields)) + " c"
        T.append(rf"\begin{{longtable}}{{@{{}}{cols}@{{}}}}")
        head = rf"\rowcolor{{rule!40}}\footnotesize\textbf{{{_tex(rng)}}} & " + \
            " & ".join(rf"\footnotesize\textbf{{{_tex(f)}}}" for f in dose_fields) + r" & \footnotesize\textbf{OK}\\"
        T.append(head)
        T.append(r"\endhead")
        for b in entry["bands"]:
            lo, hi = ((b.get("weight_kg_min"), b.get("weight_kg_max")) if key == "weight"
                      else (b.get("age_months_min"), b.get("age_months_max")))
            band = f"{lo}--{'$\\infty$' if hi is None else hi}"
            cells = " & ".join(rf"\footnotesize {_tex(b.get(f,''))}" for f in dose_fields)
            T.append(rf"\footnotesize {band} & {cells} & {ok_box(name)}\\")
        T.append(r"\end{longtable}")

    T.append(r"\section*{Specific things to check}")
    items = list(imci_dosing._tables()["_meta"]["review_checklist"]) + [
        "Fever severe label kept as very_severe_febrile_disease (2014) with bulging fontanelle added -- acceptable?",
        "HIV severity: all tiers MODERATE except hiv_infection_unlikely (MILD), none PINK (ART not urgent) -- acceptable?",
        "Young-infant treatment emits no specific dose (IM antibiotics / referral only) -- acceptable?",
    ]
    T.append(r"\begin{itemize}[leftmargin=2.4em,itemsep=5pt,label={}]")
    for it in items:
        T.append(rf"\item {ok_box('chk')}\hspace{{0.4em}}\footnotesize {_tex(it)}")
    T.append(r"\end{itemize}")
    T.append(r"{\footnotesize Full per-branch checklist items are in the "
             r"\texttt{src/sft/extended\_protocol.py} and \texttt{src/sft/young\_infant.py} docstrings.}\\[10pt]")

    T.append(r"\section*{Sign-off}")
    T.append(r"\begin{tabular}{@{}p{5.5cm}p{5.5cm}p{4.5cm}@{}}")
    fid[0] += 1; T.append(r"\TextField[name=reviewer,width=5.2cm]{} & "
                          r"\TextField[name=signature,width=5.2cm]{} & "
                          r"\TextField[name=date,width=4.2cm]{}\\")
    T.append(r"\footnotesize Reviewer name & \footnotesize Signature & \footnotesize Date\\")
    T.append(r"\end{tabular}\\[8pt]")
    T.append(r"\TextField[name=adaptation,width=\linewidth]{}\\[-2pt]")
    T.append(r"{\footnotesize National adaptation reviewed against}")

    T.append(r"\end{Form}")
    T.append(r"\end{document}")
    return "\n".join(T)


def main() -> int:
    L: list[str] = []
    L.append("# IMCI extended + young-infant branches — clinician review packet\n")
    L.append("**Status: UNREVIEWED.** These classifications and dose bands drive training data ONLY "
             "with `scripts/generate_sft_data.py --include-extended`, which is OFF by default. Nothing "
             "here ships until a qualified clinician signs it off against the current national IMCI "
             "adaptation. This packet is generated from the code (the classifiers were run to produce "
             "every row) so it cannot drift from what the model was trained to say.\n")
    L.append("**Sources:** classifications transcribed in `data/imci_2022/classifications.json` "
             "(2022 SA adaptation, cross-checked vs the WHO 2014 generic); doses in "
             "`data/imci_2022/dosing_tables.json`. Severity uses the IMCI colour tiers.\n")
    L.append("**How to review each row:** confirm the *trigger* and the *severity* against the "
             "national chart; note any correction in the last column; tick the section sign-off box. "
             "For doses, confirm each band boundary and value.\n")

    L.append("\n## Part A / B — classifications (severity + trigger + action, as implemented)\n")
    for title, cls, clf, cases in SECTIONS:
        L.append(f"\n### {title}\n")
        L.append("| Classification | Severity | Trigger (as coded) | Action | Drugs (dosed) | Clinician: OK? correction |")
        L.append("|---|---|---|---|---|---|")
        seen = set()
        for inp in cases:
            r = clf(cls(**inp))
            if r is None or r.condition_label in seen:
                continue
            seen.add(r.condition_label)
            reason = " ".join(r.reasoning).replace("|", "/")
            action = r.recommended_action.replace("|", "/")
            sev = r.classification.value
            L.append(f"| `{r.condition_label}` | {TIER.get(sev, sev)} | {reason} | {action} | "
                     f"{_drugs_for(r.condition_label)} | |")
        L.append(f"\n- [ ] **{title}** reviewed and correct  ·  corrections: ______________________\n")

    L.append("\n## Part C — dosing tables (every band)\n")
    L.append("Doses are NOT graded by the model scorer; they are emitted verbatim from these tables. "
             "Confirm every band.\n")
    tables = imci_dosing._tables()["drugs"]
    for name, entry in tables.items():
        key = entry["key"]
        L.append(f"\n### {name} — {entry.get('indication','')}  \n"
                 f"route: {entry.get('route','')}  ·  {entry.get('frequency','')}  ·  source: {entry.get('source','')}  ·  keyed by {key}")
        if entry.get("note"):
            L.append(f"  \nnote: {entry['note']}")
        if entry.get("dilution"):
            L.append(f"  \ndilution: {entry['dilution']}")
        # dose columns = whatever numeric fields the band carries
        sample = entry["bands"][0]
        dose_fields = [k for k in sample if k not in
                       ("age_band", "weight_kg_min", "weight_kg_max", "age_months_min", "age_months_max")]
        rng = "weight (kg)" if key == "weight" else "age (months)"
        L.append(f"\n| {rng} band | " + " | ".join(dose_fields) + " | OK? |")
        L.append("|---|" + "|".join(["---"] * len(dose_fields)) + "|---|")
        for b in entry["bands"]:
            if key == "weight":
                lo, hi = b.get("weight_kg_min"), b.get("weight_kg_max")
            else:
                lo, hi = b.get("age_months_min"), b.get("age_months_max")
            band = f"{lo}–{'∞' if hi is None else hi}"
            vals = " | ".join(str(b.get(f, "")) for f in dose_fields)
            L.append(f"| {band} | {vals} | |")
    L.append("\n- [ ] **All dosing tables** reviewed and correct  ·  corrections: ______________\n")

    L.append("\n## Part D — specific things to check (from the code review checklists)\n")
    for item in imci_dosing._tables()["_meta"]["review_checklist"]:
        L.append(f"- [ ] {item}")
    L.append("- [ ] Fever severe label kept as `very_severe_febrile_disease` (2014) + bulging fontanelle added — acceptable?")
    L.append("- [ ] HIV severity: all tiers MODERATE except `hiv_infection_unlikely` (MILD), none PINK "
            "(ART not urgent) — acceptable?")
    L.append("- [ ] Young-infant treatment emits NO specific dose (IM antibiotics / referral only) — acceptable?")
    L.append("\nSee `src/sft/extended_protocol.py` and `src/sft/young_infant.py` docstrings for the "
             "full per-branch checklist items (numbered 1–12 and 1–5 respectively).\n")

    L.append("\n---\nReviewer: ____________________  Signature: ____________________  Date: __________\n")

    OUT.write_text("\n".join(L))
    OUT_HTML.write_text(emit_html())
    OUT_TEX.write_text(emit_tex())
    print(f"wrote {OUT}, {OUT_HTML}, and {OUT_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
