"""
validate_and_fix.py
────────────────────
Usage:
    python validate_and_fix.py input.json [fixed_output.json]

Applies three fixes automatically:
  1. L2_Filtering_Type compound values ("X; Not_stated") → take first valid token
  2. Free-text fields missing from categorical → copy from detailed
  3. Reports any remaining issues that require manual correction
"""

import sys, json, copy

FIELDS_29 = [
    'BibKey','Paper_Type','Language_Focus','Tasks_Category','Dataset_Size_Scale',
    'Domains_Category','Downstream_Tasks_Types','L1_Data_Sourcing_Type',
    'L2_Filtering_Type','L3_Task_Typing_Type','L4_Templating_Type',
    'L5_Generation_Strategy','L6_Validation_Type','L7_Storage_Type',
    'L8_Generalisability_Type','Declared_Usage','Blind_Test','Decontamination',
    'Model_Teacher_Type','Model_Student_Type','Metrics_Standard_Type',
    'Metrics_LLM_Based_Type','Formal_Notation_Present','Arabic_Markers_Present',
    'Arabic_Markers_Detail','Distractors_Used','Other_Notable_Method_Choices',
    'Method_Summary','Paper_Method_Tags'
]

FREE_TEXT_FIELDS = ['Arabic_Markers_Detail','Other_Notable_Method_Choices','Method_Summary']

CONTROLLED = {
    'Paper_Type':            ['Benchmark','Leaderboard','Survey-Position','Instruction-Tuning','Synthetic-Pipeline','Tokenization-Tool','Evaluation-Methodology','Other'],
    'Language_Focus':        ['Arabic','Multilingual_incl_Arabic','Non-Arabic_low-resource','English-only','Multilingual_no_Arabic'],
    'Tasks_Category':        ['Single-task','Multi-task','Survey','Not_applicable'],
    'Dataset_Size_Scale':    ['10^3_or_less','10^4','10^5','10^6_or_more','Not_applicable','Not_stated'],
    'Domains_Category':      ['General','Domain-specific','Mixed','Survey','Not_applicable'],
    'L1_Data_Sourcing_Type': ['Human-seed-only','Synthetic-seed-only','Native-corpus','Web-crawl','KB','Mixed','Other'],
    'L2_Filtering_Type':     ['Rules','Lexical-similarity','Lexical-similarity+Classifier','LLM-scoring','Agreement-based','Human','None','Not_stated'],
    'L3_Task_Typing_Type':   ['No-formal-taxonomy','Benchmark-style-taxonomy','Complexity-based','Single-task','Other'],
    'L4_Templating_Type':    ['Single-turn-unstructured','Single-turn-evolution-prompts','Multi-turn-dialog','Structured-tuples','Symbolic-templates','Not_stated'],
    'L5_Generation_Strategy':['Self-Instruct','Evol-Instruct','Few-shot-prompting','Retrieval-QA','KB-driven','Tagged-corruption','Collaborative-SLM','Self-Improving-Judge','Human-only','None (curation-only)','Other'],
    'L6_Validation_Type':    ['Automatic','Automatic+Downstream','LLM-judge','LLM-judge+Downstream','Human','None','Not_stated'],
    'L7_Storage_Type':       ['Flat-JSON-no-splits','Structured-benchmark-schema','Dynamic-leaderboard','Not_stated'],
    'L8_Generalisability_Type':['Single-language-no-transfer','Multilingual-claimed','Low-resource-transfer-claimed','Not_stated'],
    'Declared_Usage':        ['Training','Benchmark','Both','Ambiguous'],
    'Blind_Test':            ['Yes','No','Not_stated'],
    'Decontamination':       ['Yes','Partial','No','Not_stated'],
    'Model_Teacher_Type':    ['Proprietary-LLM','Open-LLM','None','Not_stated'],
    'Model_Student_Type':    ['Fine-tuned-LLM','Benchmark-only','Tokenizer-only','None','Not_stated'],
    'Metrics_Standard_Type': ['Task-metrics','Tokenization-metrics','None','Not_stated'],
    'Metrics_LLM_Based_Type':['LLM-as-judge','None','Not_stated'],
    'Formal_Notation_Present':['Algorithmic-pseudo-code','Mathematical-equations','Both','None','Not_stated'],
    'Arabic_Markers_Present':['Yes','No'],
    'Distractors_Used':      ['Yes','No','Not_stated'],
}

def fix_compound_value(field, value):
    """'Human; Not_stated' → 'Human'  (take first valid token)"""
    if field not in CONTROLLED:
        return value, False
    valid = CONTROLLED[field]
    if value in valid:
        return value, False
    # Try splitting on '; ' or ','
    for sep in ['; ', ';', ', ', ',']:
        parts = [p.strip() for p in value.split(sep)]
        for p in parts:
            if p in valid:
                return p, True
    return value, False  # unfixable


def validate_fix(data):
    fixed = copy.deepcopy(data)
    report = {}

    for bibkey, paper in fixed.items():
        report[bibkey] = {'auto_fixed': [], 'remaining_issues': []}

        for view in ['detailed', 'categorical']:
            vdata = paper.get(view, {})

            # Fix 1: compound controlled values
            for field, valid_opts in CONTROLLED.items():
                val = vdata.get(field)
                if val and val not in valid_opts:
                    new_val, was_fixed = fix_compound_value(field, val)
                    if was_fixed:
                        vdata[field] = new_val
                        report[bibkey]['auto_fixed'].append(
                            f'{view}/{field}: "{val}" → "{new_val}"')
                    else:
                        report[bibkey]['remaining_issues'].append(
                            f'{view}/{field}: "{val}" not in valid set')

            # Fix 2: free-text fields missing from categorical → copy from detailed
            if view == 'categorical':
                for ft in FREE_TEXT_FIELDS:
                    if ft not in vdata:
                        src = paper.get('detailed', {}).get(ft, 'Not_stated')
                        vdata[ft] = src
                        report[bibkey]['auto_fixed'].append(
                            f'categorical/{ft}: copied from detailed')

            # Fix 3: missing fields → set Not_stated
            for f in FIELDS_29:
                if f not in vdata:
                    vdata[f] = 'Not_stated'
                    report[bibkey]['remaining_issues'].append(
                        f'{view}/{f}: was missing → set to Not_stated')

    return fixed, report


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_and_fix.py input.json [output.json]")
        sys.exit(1)

    in_path  = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else in_path.replace('.json', '_fixed.json')

    with open(in_path, encoding='utf-8') as f:
        data = json.load(f)

    fixed, report = validate_fix(data)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(f'\n✓ Fixed JSON saved → {out_path}\n')
    print('=== REPORT ===')
    all_clean = True
    for bibkey, r in report.items():
        print(f'\n[{bibkey}]')
        if r['auto_fixed']:
            for msg in r['auto_fixed']:
                print(f'  AUTO-FIXED : {msg}')
        if r['remaining_issues']:
            all_clean = False
            for msg in r['remaining_issues']:
                print(f'  ⚠ MANUAL   : {msg}')
        if not r['auto_fixed'] and not r['remaining_issues']:
            print('  ✓ Clean — no issues')

    if all_clean:
        print('\n✓ All papers clean after auto-fix. Ready for json_to_excel.py')
    else:
        print('\n⚠ Some issues require manual review before running json_to_excel.py')


if __name__ == '__main__':
    main()
