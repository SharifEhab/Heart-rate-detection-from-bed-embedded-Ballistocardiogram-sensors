"""
Run the BCG and RR signal processing pipeline

This script processes paired BCG and RR signal files by:
1. Synchronizing BCG and RR signals based on timestamps
2. Detecting and removing body movement segments 
3. Interpolating RR data to match the BCG time grid
4. Windowing the filtered signals for further analysis
"""
"""
This script was for testing 
Still not 100% correct
"""


import os
import argparse
import numpy as np
from process_signals import process_pair, parse_filename

def main():
    parser = argparse.ArgumentParser(description='Process BCG and RR signal pairs')
    parser.add_argument('--data_dir', default='D:/Data Analytics/projectrepo/dataset/data', 
                        help='Path to data directory')
    parser.add_argument('--output_dir', default='../results',
                        help='Path to save results')
    parser.add_argument('--subject', default=None, type=str,
                        help='Process specific subject (e.g., "01")')
    parser.add_argument('--date', default=None, type=str,
                        help='Process specific date (e.g., "20231104")')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots for movement detection')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Discover all BCG & RR pairs
    pairs = []  # list of (subj, date, bcg_path, rr_path)
    
    for subj in sorted(os.listdir(args.data_dir)):
        # Skip if filtering by subject and this isn't the requested one
        if args.subject and subj != args.subject:
            continue
            
        bdir = os.path.join(args.data_dir, subj, "BCG")
        rdir = os.path.join(args.data_dir, subj, "Reference", "RR")
        
        if not os.path.isdir(bdir):
            continue
            
        bcg_files = [f for f in os.listdir(bdir) if f.endswith("_BCG.csv")]
        rr_files = os.path.isdir(rdir) and [f for f in os.listdir(rdir) if f.endswith("_RR.csv")] or []
        
        for bfn in bcg_files:
            subj_, date, kind = parse_filename(bfn)
            
            # Skip if filtering by date and this isn't the requested one
            if args.date and date != args.date:
                continue
                
            # Find matching RR file
            match_rr = None
            for rfn in rr_files:
                if parse_filename(rfn)[1] == date:
                    match_rr = os.path.join(rdir, rfn)
            
            if match_rr:  # Only add pairs that have both BCG and RR
                pairs.append((subj, date, os.path.join(bdir, bfn), match_rr))
    
    print(f"Found {len(pairs)} BCG-RR pairs to process.")
    
    # Process each pair
    results = {}
    for i, (subj, date, bcg_path, rr_path) in enumerate(pairs):
        print(f"Processing pair {i+1}/{len(pairs)}: Subject {subj}, Date {date}")
        
        # Process this pair
        try:
            bcg_windows, rr_windows, time_windows = process_pair(
                subj, date, bcg_path, rr_path, args.output_dir
            )
            results[(subj, date)] = {
                'bcg_windows': bcg_windows,
                'rr_windows': rr_windows,
                'time_windows': time_windows,
                'num_windows': len(bcg_windows)
            }
            print(f"  Successfully processed {len(bcg_windows)} windows")
        except Exception as e:
            print(f"  Error processing: {e}")
    
    # Print summary
    print("\nProcessing Summary:")
    print("-------------------")
    total_windows = sum(r['num_windows'] for r in results.values())
    print(f"Total subjects: {len(set(subj for subj, _ in results.keys()))}")
    print(f"Total recordings: {len(results)}")
    print(f"Total windows: {total_windows}")
    
    print("\nWindows per subject-date:")
    for (subj, date), r in results.items():
        print(f"  Subject {subj}, Date {date}: {r['num_windows']} windows")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main() 