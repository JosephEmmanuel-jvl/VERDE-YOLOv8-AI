"""Persist Codex visual findings without changing historical labels or assigning training labels."""
import csv, json, re
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent; E=HERE/'evidence'; P=E/'new_source_pilot'

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

pilot=read(P/'EMBEDDED_RGB_LINKAGE.csv')
protected={r['sha256'] for r in read(E/'PROTECTED_INPUT_CHECKSUMS_BEFORE.csv')}
for r in pilot:
    normal=r['filename'].startswith('Normal');number=int(r['filename'].split('_')[-1].split('.')[0])
    r.update({'reviewer':'CODEX_VISUAL_REVIEW_ONLY','review_date':'2026-10-24','source_treatment':'IRRIGATED' if normal else 'NON_IRRIGATED','visual_phenotype':'GREEN_TURGID_APPEARANCE' if normal or number<=6 else 'POSSIBLE_PROGRESSIVE_LOSS_OF_TURGOR','visual_class':'UNASSIGNED_HELD','decision':'HOLD_GROWING_PLANT_CONTEXT','reason':'Whole lettuce head visible on tabletop; roots/container/growing linkage not visually established. Treatment alone is not a visual class label.','license_confirmed':'YES_CC_BY_4_0','publisher_original_hash_confirmed':'YES','source_binary_linkage':'PUBLISHER_SHA256_AND_EMBEDDED_JPEG_BYTE_OFFSETS','exact_overlap_protected_local_inputs':'YES' if r['original_sha256'] in protected or r['embedded_sha256'] in protected else 'NO','derivative_screen':'REPEATED_TWO_PLANT_SEQUENCE; broader image-content exclusion review incomplete','new_approved_training_image':'NO','assignment':'SOURCE_QUARANTINE_NO_TRAIN_VAL_TEST; prospective acceptance candidate only','acceptance_qualified':'NO','human_supervisor_review':'NOT_PERFORMED'})
write(P/'CODEX_VISUAL_REVIEW.csv',pilot)

samples=read(E/'HISTORICAL_LABEL_REVIEW_SAMPLE.csv')
for r in samples:
    cid=r['candidate_id'];cls=r['class'];findings=[]
    if 'AGRIBOT' in cid:
        findings=['GROWING_FIELD_PLANTS_VISIBLE','SEPARATE_SMALL_PLANT_BOXES_PRESENT','CHECK_OUTER_FOLIAGE_TIGHTNESS_AND_SMALL_EDGE_TARGETS']
    elif cls=='Healthy':
        findings=['VISIBLE_GROWING_LETTUCE','PARTIAL_OR_CLOSE_CROP','FOLIAGE_MAY_TOUCH_IMAGE_EDGE']
    elif cls=='Yellowing' and cid.startswith('RAW-'):
        findings=['PALE_LEAVES_VISIBLE','COLOUR_CAST_OR_EXPOSURE_CAN_CONFOUND_YELLOWING','GROWING_CONTAINER_LINKAGE_REQUIRES_REVIEW']
    elif cls=='Yellowing' and 'V006' in cid:
        findings=['LEAF_DOMINATED_CLOSE_CROP','MOTTLED_COLOUR_VISIBLE','WHOLE_PLANT_BOUNDARY_NOT_FULLY_SHOWN']
    elif cls=='Yellowing':
        findings=['VISIBLE_CHLOROSIS','GROWING_HYDROPONIC_PLANT','CHECK_YELLOWING_VS_WILTING_OVERLAP']
    elif cls=='Wilting' and cid.startswith('RAW-'):
        findings=['POTTED_PLANT_VISIBLE','DROOPING_OR_SPARSE_STRESSED_FOLIAGE','YELLOWING_ALSO_PRESENT']
    else:
        findings=['HYDROPONIC_LETTUCE_VISIBLE','DOMINANT_CHLOROSIS_OR_DAMAGE','STRUCTURAL_WILT_NEEDS_HUMAN_CONFIRMATION']
    r.update({'reviewer':'CODEX_ONLY','review_scope':'18_IMAGE_STRATIFIED_SAMPLE_NOT_FULL_DATASET_APPROVAL','findings':';'.join(findings),'decision':'HISTORICAL_LABEL_UNCHANGED_REVIEW_FLAG_ONLY'})
write(E/'HISTORICAL_LABEL_VISUAL_FINDINGS.csv',samples)

records=read(E/'EXISTING_CONDITION_DATA_MANIFEST.csv')
families=defaultdict(list)
for r in records:
    m=re.match(r'(D1-V\d+_.+_\d{8})_\d+_S\d+_\d+\.jpg',r['candidate_id'])
    if m:families[m.group(1)].append(r)
warnings=[]
for k,rs in families.items():
    splits=sorted({r['split'] for r in rs})
    if len(splits)>1:warnings.append({'inferred_filename_family':k,'splits':';'.join(splits),'images':len(rs),'candidate_ids':';'.join(r['candidate_id'] for r in rs),'status':'POTENTIAL_SHARED_CAPTURE_SEQUENCE_NOT_CONFIRMED','basis':'Same encoded source/date prefix, different frame numbers; source identity needs verification','action':'No historical split or label changed'})
write(E/'POTENTIAL_FILENAME_SEQUENCE_OVERLAP.csv',warnings or [{'inferred_filename_family':'NONE','splits':'','images':0,'candidate_ids':'','status':'NONE_DETECTED','basis':'Filename heuristic only','action':'No historical split changed'}])

summary={'pilot_originals':len(pilot),'embedded_rgb':len(pilot),'plant_groups':2,'training_approved':0,'quarantine_holds':len(pilot),'historical_visual_review_sample':len(samples),'potential_filename_sequence_cross_split_families':len(warnings),'independent_qualified_acceptance_images':0,'human_supervisor_review_performed':False}
(E/'VISUAL_REVIEW_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
