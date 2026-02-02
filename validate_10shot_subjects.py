#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NOTE: This Python script is heavily commented to clarify intent and execution flow.
# -----------------------------------------------------------------------------

"""
Validate 10-shot PDGM subjects for consistency.

Checks:
- All required modalities present (t1, t2, flair, mask)
- Image shapes are consistent
- Intensity ranges are reasonable
- No corrupted files
- Masks have reasonable tumor volumes
"""

# Import dependencies used by this module.
import os
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
from tabulate import tabulate


# Function: `check_subject` implements a reusable processing step.
def check_subject(subject_root: Path, subject_id: str):
    """Check a single subject for consistency."""
    results = {
        'subject_id': subject_id,
        'exists': False,
        't1_found': False,
        't2_found': False,
        'flair_found': False,
        'mask_found': False,
        't1_shape': None,
        't2_shape': None,
        'flair_shape': None,
        'mask_shape': None,
        't1_range': None,
        't2_range': None,
        'flair_range': None,
        'mask_voxels': None,
        'mask_percent': None,
        'status': 'OK',
        'issues': []
    }
    
    subject_dir = subject_root / subject_id
    
    # Control-flow branch for conditional or iterative execution.
    if not subject_dir.exists():
        results['exists'] = False
        results['status'] = 'ERROR'
        results['issues'].append('Directory not found')
        # Return the computed value to the caller.
        return results
    
    results['exists'] = True
    
    # Find modality files
    files = list(subject_dir.glob('*.nii.gz'))
    
    # Function: `find_file` implements a reusable processing step.
    def find_file(keyword):
        """Find file containing keyword (case insensitive)."""
        # Control-flow branch for conditional or iterative execution.
        for f in files:
            # Control-flow branch for conditional or iterative execution.
            if keyword in f.name.lower():
                # Return the computed value to the caller.
                return f
        # Return the computed value to the caller.
        return None
    
    # Check each modality
    modalities = {
        't1': find_file('t1'),
        't2': find_file('t2'),
        'flair': find_file('flair'),
        'mask': find_file('mask')
    }
    
    # Control-flow branch for conditional or iterative execution.
    for mod, filepath in modalities.items():
        # Control-flow branch for conditional or iterative execution.
        if filepath:
            results[f'{mod}_found'] = True
            
            # Control-flow branch for conditional or iterative execution.
            try:
                # Load and check
                img = nib.load(filepath)
                data = img.get_fdata()
                
                # Store shape
                results[f'{mod}_shape'] = data.shape
                
                # For image modalities, check intensity range
                # Control-flow branch for conditional or iterative execution.
                if mod != 'mask':
                    # Get non-zero values only
                    nonzero = data[data > 0]
                    # Control-flow branch for conditional or iterative execution.
                    if len(nonzero) > 0:
                        results[f'{mod}_range'] = (float(nonzero.min()), float(nonzero.max()))
                    else:
                        results[f'{mod}_range'] = (0.0, 0.0)
                        results['issues'].append(f'{mod} is all zeros')
                        results['status'] = 'WARNING'
                
                # For mask, check tumor volume
                # Control-flow branch for conditional or iterative execution.
                if mod == 'mask':
                    mask_binary = data > 0
                    voxels = int(mask_binary.sum())
                    total_voxels = int(np.prod(data.shape))
                    percent = (voxels / total_voxels * 100) if total_voxels > 0 else 0
                    
                    results['mask_voxels'] = voxels
                    results['mask_percent'] = percent
                    
                    # Control-flow branch for conditional or iterative execution.
                    if voxels == 0:
                        results['issues'].append('Mask is empty')
                        results['status'] = 'ERROR'
                    # Control-flow branch for conditional or iterative execution.
                    elif percent < 0.01:
                        results['issues'].append(f'Mask very small ({percent:.4f}%)')
                        results['status'] = 'WARNING'
                    # Control-flow branch for conditional or iterative execution.
                    elif percent > 50:
                        results['issues'].append(f'Mask very large ({percent:.2f}%)')
                        results['status'] = 'WARNING'
                
            # Control-flow branch for conditional or iterative execution.
            except Exception as e:
                results['issues'].append(f'Error loading {mod}: {str(e)}')
                results['status'] = 'ERROR'
    
    # Check for missing modalities
    # Control-flow branch for conditional or iterative execution.
    for mod in ['t1', 't2', 'flair', 'mask']:
        # Control-flow branch for conditional or iterative execution.
        if not results[f'{mod}_found']:
            results['issues'].append(f'Missing {mod}')
            results['status'] = 'ERROR'
    
    # Check shape consistency across modalities
    # Control-flow branch for conditional or iterative execution.
    if results['t1_found'] and results['t2_found'] and results['flair_found']:
        shapes = [results['t1_shape'], results['t2_shape'], results['flair_shape']]
        # Control-flow branch for conditional or iterative execution.
        if results['mask_found']:
            shapes.append(results['mask_shape'])
        
        # Control-flow branch for conditional or iterative execution.
        if len(set(shapes)) > 1:
            results['issues'].append(f'Shape mismatch: {shapes}')
            results['status'] = 'ERROR'
    
    # Return the computed value to the caller.
    return results


# Function: `main` implements a reusable processing step.
def main():
    # Import dependencies used by this module.
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate 10-shot PDGM subjects')
    parser.add_argument('--pdgm_root', required=True, help='PDGM data root directory')
    parser.add_argument('--fewshot', required=True, help='Few-shot subject list file')
    args = parser.parse_args()
    
    pdgm_root = Path(args.pdgm_root)
    fewshot_file = Path(args.fewshot)
    
    # Control-flow branch for conditional or iterative execution.
    if not pdgm_root.exists():
        print(f"ERROR: PDGM root not found: {pdgm_root}")
        sys.exit(1)
    
    # Control-flow branch for conditional or iterative execution.
    if not fewshot_file.exists():
        print(f"ERROR: Fewshot file not found: {fewshot_file}")
        sys.exit(1)
    
    # Read subject list
    # Control-flow branch for conditional or iterative execution.
    with open(fewshot_file, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]
    
    print("="*80)
    print("10-SHOT PDGM SUBJECT VALIDATION")
    print("="*80)
    print(f"PDGM Root: {pdgm_root}")
    print(f"Fewshot File: {fewshot_file}")
    print(f"Number of subjects: {len(subjects)}")
    print()
    
    # Control-flow branch for conditional or iterative execution.
    if len(subjects) != 10:
        print(f"WARNING: Expected 10 subjects, found {len(subjects)}")
        print()
    
    # Check each subject
    all_results = []
    # Control-flow branch for conditional or iterative execution.
    for subject_id in subjects:
        print(f"Checking {subject_id}...", end=' ')
        result = check_subject(pdgm_root, subject_id)
        all_results.append(result)
        print(result['status'])
    
    print()
    print("="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    # Summary table
    table_data = []
    # Control-flow branch for conditional or iterative execution.
    for r in all_results:
        table_data.append([
            r['subject_id'],
            r['exists'],
            r['t1_found'],
            r['t2_found'],
            r['flair_found'],
            r['mask_found'],
            r['mask_voxels'] if r['mask_voxels'] else 'N/A',
            f"{r['mask_percent']:.2f}%" if r['mask_percent'] else 'N/A',
            r['status']
        ])
    
    headers = ['Subject', 'Exists', 'T1', 'T2', 'FLAIR', 'Mask', 'Mask Voxels', 'Mask %', 'Status']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print()
    print("="*80)
    print("SHAPE CONSISTENCY")
    print("="*80)
    
    # Check shape consistency across subjects
    shapes = []
    # Control-flow branch for conditional or iterative execution.
    for r in all_results:
        # Control-flow branch for conditional or iterative execution.
        if r['t1_shape']:
            shapes.append(r['t1_shape'])
    
    # Control-flow branch for conditional or iterative execution.
    if len(set(shapes)) == 1:
        print(f"✓ All subjects have consistent shape: {shapes[0]}")
    else:
        print("✗ Shape inconsistency detected!")
        # Control-flow branch for conditional or iterative execution.
        for r in all_results:
            # Control-flow branch for conditional or iterative execution.
            if r['t1_shape']:
                print(f"  {r['subject_id']}: {r['t1_shape']}")
    
    print()
    print("="*80)
    print("INTENSITY RANGES (non-zero voxels)")
    print("="*80)
    
    # Show intensity ranges
    # Control-flow branch for conditional or iterative execution.
    for mod in ['t1', 't2', 'flair']:
        print(f"\n{mod.upper()}:")
        table_data = []
        # Control-flow branch for conditional or iterative execution.
        for r in all_results:
            # Control-flow branch for conditional or iterative execution.
            if r[f'{mod}_range']:
                min_val, max_val = r[f'{mod}_range']
                table_data.append([
                    r['subject_id'],
                    f"{min_val:.2f}",
                    f"{max_val:.2f}",
                    f"{max_val - min_val:.2f}"
                ])
        
        # Control-flow branch for conditional or iterative execution.
        if table_data:
            print(tabulate(table_data, headers=['Subject', 'Min', 'Max', 'Range'], tablefmt='simple'))
    
    print()
    print("="*80)
    print("MASK STATISTICS")
    print("="*80)
    
    mask_voxels = [r['mask_voxels'] for r in all_results if r['mask_voxels']]
    mask_percents = [r['mask_percent'] for r in all_results if r['mask_percent']]
    
    # Control-flow branch for conditional or iterative execution.
    if mask_voxels:
        print(f"Mask Voxels:")
        print(f"  Min:    {min(mask_voxels):,}")
        print(f"  Max:    {max(mask_voxels):,}")
        print(f"  Mean:   {np.mean(mask_voxels):,.0f}")
        print(f"  Median: {np.median(mask_voxels):,.0f}")
        print(f"  Std:    {np.std(mask_voxels):,.0f}")
        print()
        print(f"Mask Percentage:")
        print(f"  Min:    {min(mask_percents):.4f}%")
        print(f"  Max:    {max(mask_percents):.4f}%")
        print(f"  Mean:   {np.mean(mask_percents):.4f}%")
        print(f"  Median: {np.median(mask_percents):.4f}%")
    
    print()
    print("="*80)
    print("ISSUES FOUND")
    print("="*80)
    
    has_issues = False
    # Control-flow branch for conditional or iterative execution.
    for r in all_results:
        # Control-flow branch for conditional or iterative execution.
        if r['issues']:
            has_issues = True
            print(f"\n{r['subject_id']}:")
            # Control-flow branch for conditional or iterative execution.
            for issue in r['issues']:
                print(f"  • {issue}")
    
    # Control-flow branch for conditional or iterative execution.
    if not has_issues:
        print("✓ No issues found!")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    ok_count = sum(1 for r in all_results if r['status'] == 'OK')
    warning_count = sum(1 for r in all_results if r['status'] == 'WARNING')
    error_count = sum(1 for r in all_results if r['status'] == 'ERROR')
    
    print(f"Total subjects: {len(all_results)}")
    print(f"  OK:       {ok_count}")
    print(f"  Warnings: {warning_count}")
    print(f"  Errors:   {error_count}")
    print()
    
    # Control-flow branch for conditional or iterative execution.
    if error_count > 0:
        print("⚠️  ERRORS DETECTED - Fix before training!")
        sys.exit(1)
    # Control-flow branch for conditional or iterative execution.
    elif warning_count > 0:
        print("⚠️  WARNINGS DETECTED - Review before training")
        sys.exit(0)
    else:
        print("✓ ALL SUBJECTS VALIDATED SUCCESSFULLY")
        print("✓ Ready for 10-shot training!")
        sys.exit(0)


# Run the CLI entry point when this file is executed directly.
# Control-flow branch for conditional or iterative execution.
if __name__ == '__main__':
    main()