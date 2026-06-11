"""
Batch Korotkoff Validation Script
Runs cross-validation on multiple pairs of RF and Stethoscope recordings.
"""
import subprocess
import os

pairs = [
    {
        'rf': r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_sthe_1.h5',
        'aud': r'd:\Bioview\My_RF_work_v1\data_new\korotoff_audio_stethoscope2.mp4',
        'out': r'd:\Bioview\My_RF_work_v1\data_new\koro_rf_vs_stethoscope_pair1.png'
    },
    {
        'rf': r'd:\Bioview\My_RF_work_v1\data_new\rec_koro_sthe_2.h5',
        'aud': r'd:\Bioview\My_RF_work_v1\data_new\korotoff_audio_stethoscope3.mp4',
        'out': r'd:\Bioview\My_RF_work_v1\data_new\koro_rf_vs_stethoscope_pair2.png'
    }
]

python_exe = r'D:\Bioview\bioview_env\Scripts\python.exe'
template_script = r'd:\Bioview\My_RF_work_v1\scratch\koro_rf_vs_stethoscope.py'

for i, pair in enumerate(pairs):
    print(f"\nProcessing Pair {i+1}...")
    print(f"  RF: {os.path.basename(pair['rf'])}")
    print(f"  Audio: {os.path.basename(pair['aud'])}")
    
    # Read the template script
    with open(template_script, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace the paths
    # We need to find the specific lines in the template
    # RF_PATH    = r'...'
    # AUDIO_PATH = r'...'
    # OUTPUT_IMG = r'...'
    
    new_content = content
    # Find and replace the config section
    import re
    def safe_sub(pattern, repl, string):
        # Escape backslashes for the replacement string in re.sub
        safe_repl = repl.replace('\\', '\\\\')
        return re.sub(pattern, safe_repl, string)

    new_content = safe_sub(r"RF_PATH\s*=\s*r'.*?'", f"RF_PATH = r'{pair['rf']}'", new_content)
    new_content = safe_sub(r"AUDIO_PATH\s*=\s*r'.*?'", f"AUDIO_PATH = r'{pair['aud']}'", new_content)
    new_content = safe_sub(r"OUTPUT_IMG\s*=\s*r'.*?'", f"OUTPUT_IMG = r'{pair['out']}'", new_content)
    
    # Write to a temporary script
    temp_script = f"temp_analysis_{i+1}.py"
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    # Run the script
    print(f"  Running analysis...")
    try:
        result = subprocess.run([python_exe, temp_script], capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error processing pair {i+1}:")
        print(e.stderr)
    
    # Clean up
    if os.path.exists(temp_script):
        os.remove(temp_script)

print("\nBatch processing complete.")
