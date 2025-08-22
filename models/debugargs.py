#!/usr/bin/env python3
"""
DeltaTwin Argument Debugging Script
This script helps identify how arguments are passed by DeltaTwin
"""

import os
import sys
import json
import datetime

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

def log_with_timestamp(message, use_stderr=False):
    """Log with timestamp for visibility"""
    timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    formatted_msg = f"[{timestamp}] {message}"
    
    if use_stderr:
        print(formatted_msg, file=sys.stderr, flush=True)
    else:
        print(formatted_msg, file=sys.stdout, flush=True)

def main():
    log_with_timestamp("=== DELTATWIN ARGUMENT DEBUGGING START ===")
    
    # Basic system info
    log_with_timestamp(f"Python version: {sys.version}")
    log_with_timestamp(f"Working directory: {os.getcwd()}")
    log_with_timestamp(f"Script name: {sys.argv[0] if sys.argv else 'UNKNOWN'}")
    
    # Detailed argument analysis
    log_with_timestamp(f"Total arguments received: {len(sys.argv)}")
    log_with_timestamp("Raw sys.argv content:")
    
    for i, arg in enumerate(sys.argv):
        log_with_timestamp(f"  argv[{i}]: '{arg}' (type: {type(arg)}, length: {len(str(arg))})")
    
    # Environment variable inspection
    log_with_timestamp("Checking relevant environment variables:")
    env_vars_to_check = [
        'PYTHONUNBUFFERED',
        'PWD', 'HOME', 'PATH',
        'CDSE_KEY', 'CDSE_SECRET', 'EARTH_DATA_HUB_KEY',  # In case they're in env
        'BBOX', 'SAMPLED_FRACTION'
    ]
    
    for var in env_vars_to_check:
        value = os.environ.get(var, 'NOT_SET')
        if value != 'NOT_SET' and len(value) > 50:
            value = value[:47] + "..."
        log_with_timestamp(f"  ENV {var}: {value}")
    
    # Try to detect argument pattern
    log_with_timestamp("=== ARGUMENT PATTERN ANALYSIS ===")
    
    if len(sys.argv) >= 6:
        log_with_timestamp("Attempting to parse as expected format:")
        try:
            potential_key = sys.argv[1]
            potential_secret = sys.argv[2] 
            potential_token = sys.argv[3]
            potential_bbox = sys.argv[4]
            potential_fraction = sys.argv[5]
            
            log_with_timestamp(f"Potential CDSE Key: '{potential_key[:10]}...' (length: {len(potential_key)})")
            log_with_timestamp(f"Potential CDSE Secret: '{potential_secret[:10]}...' (length: {len(potential_secret)})")
            log_with_timestamp(f"Potential EH Token: '{potential_token[:10]}...' (length: {len(potential_token)})")
            log_with_timestamp(f"Potential BBOX: '{potential_bbox}'")
            log_with_timestamp(f"Potential Fraction: '{potential_fraction}'")
            
            # Try to validate bbox format
            try:
                bbox_coords = [float(x.strip()) for x in potential_bbox.split(',')]
                if len(bbox_coords) == 4:
                    log_with_timestamp(f"BBOX appears valid: {bbox_coords}")
                else:
                    log_with_timestamp(f"BBOX has wrong number of coordinates: {len(bbox_coords)}")
            except Exception as e:
                log_with_timestamp(f"BBOX parsing failed: {e}")
            
            # Try to validate fraction
            try:
                fraction_val = float(potential_fraction)
                log_with_timestamp(f"Fraction appears valid: {fraction_val}")
            except Exception as e:
                log_with_timestamp(f"Fraction parsing failed: {e}")
                
        except Exception as e:
            log_with_timestamp(f"Error in pattern analysis: {e}")
    else:
        log_with_timestamp(f"Insufficient arguments for expected pattern (need 6, got {len(sys.argv)})")
    
    # Check for alternative argument patterns
    log_with_timestamp("=== CHECKING ALTERNATIVE PATTERNS ===")
    
    # Look for key-value pairs
    for i, arg in enumerate(sys.argv[1:], 1):
        if '=' in arg:
            key, value = arg.split('=', 1)
            log_with_timestamp(f"Found key-value pair: {key}={value[:20]}...")
        elif arg.startswith('--'):
            log_with_timestamp(f"Found flag-style argument: {arg}")
        elif ',' in arg and len(arg.split(',')) == 4:
            log_with_timestamp(f"Found potential BBOX at position {i}: {arg}")
        elif arg.replace('.', '').isdigit():
            log_with_timestamp(f"Found potential numeric value at position {i}: {arg}")
    
    # Summary and recommendations
    log_with_timestamp("=== DEBUGGING SUMMARY ===")
    
    if len(sys.argv) < 6:
        log_with_timestamp("ISSUE: Not enough arguments provided")
        log_with_timestamp("Expected: python script.py <cdse_key> <cdse_secret> <eh_token> <bbox> <fraction>")
    elif len(sys.argv) == 6:
        log_with_timestamp("Argument count matches expected pattern")
    else:
        log_with_timestamp(f"WARNING: More arguments than expected ({len(sys.argv)} vs 6)")
    
    # Test output visibility
    log_with_timestamp("=== OUTPUT VISIBILITY TEST ===")
    for i in range(3):
        log_with_timestamp(f"STDOUT visibility test {i+1}")
        log_with_timestamp(f"STDERR visibility test {i+1}", use_stderr=True)
    
    log_with_timestamp("=== DELTATWIN ARGUMENT DEBUGGING COMPLETE ===")
    
    # Exit successfully to indicate the container can run
    sys.exit(0)

if __name__ == "__main__":
    print("IMMEDIATE OUTPUT: Script starting", flush=True)
    main()