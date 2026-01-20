import os
import subprocess
import sys
import glob

def run_tests(test_dir):
    test_files = sorted(glob.glob(os.path.join(test_dir, "test_*")))
    # Filter out files that are not executable or have extensions (like .cpp, .o)
    # On linux/mac executables usually have no extension or .app (but here we see no extension)
    test_executables = [f for f in test_files if os.access(f, os.X_OK) and not f.endswith(".cpp") and not f.endswith(".txt") and not f.endswith(".cmake")]
    
    results = []
    
    print(f"Found {len(test_executables)} test executables.")
    print("-" * 80)
    
    for test_exe in test_executables:
        test_name = os.path.basename(test_exe)
        print(f"Running {test_name}...", end="", flush=True)
        
        try:
            # QtTest usually outputs to stdout
            proc = subprocess.run([test_exe], capture_output=True, text=True, timeout=60)
            output = proc.stdout + proc.stderr
            
            # Parse output for totals
            # Totals: 10 passed, 0 failed, 0 skipped, 0 blacklisted, 6ms
            passed = False
            totals_line = ""
            for line in output.splitlines():
                if "Totals:" in line:
                    totals_line = line.strip()
                    if "failed, 0 skipped" in line or (", 0 failed" in line): 
                        # Check logic carefully. Usually: "Totals: X passed, Y failed, ..."
                        parts = line.split(',')
                        for part in parts:
                            if "failed" in part:
                                num_failed = int(part.strip().split()[0])
                                if num_failed == 0:
                                    passed = True
                                else:
                                    passed = False
                                break
                    break
            
            # Fallback check if totals line not found (e.g. crash)
            if proc.returncode != 0:
                passed = False
                
            status = "PASS" if passed else "FAIL"
            print(f" {status}")
            if not passed:
                print(f"Output:\n{output}\n")
                
            results.append({
                "name": test_name,
                "status": status,
                "totals": totals_line,
                "output": output if not passed else ""
            })
            
        except subprocess.TimeoutExpired:
            print(" TIMEOUT")
            results.append({
                "name": test_name,
                "status": "TIMEOUT",
                "totals": "",
                "output": "Test timed out after 60s"
            })
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({
                "name": test_name,
                "status": "ERROR",
                "totals": "",
                "output": str(e)
            })

    print("-" * 80)
    print("Summary:")
    passed_count = sum(1 for r in results if r["status"] == "PASS")
    print(f"Total: {len(results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {len(results) - passed_count}")
    
    # Write detailed report
    with open("test_report.md", "w") as f:
        f.write("# Unit Test Report\n\n")
        f.write(f"**Total**: {len(results)}\n")
        f.write(f"**Passed**: {passed_count}\n")
        f.write(f"**Failed**: {len(results) - passed_count}\n\n")
        
        f.write("## Detailed Results\n\n")
        f.write("| Test Name | Status | Details |\n")
        f.write("| --- | --- | --- |\n")
        for r in results:
            f.write(f"| {r['name']} | {r['status']} | {r['totals']} |\n")
            
        if len(results) - passed_count > 0:
            f.write("\n## Failures\n\n")
            for r in results:
                if r["status"] != "PASS":
                    f.write(f"### {r['name']}\n")
                    f.write("```\n")
                    f.write(r['output'])
                    f.write("\n```\n")

if __name__ == "__main__":
    run_tests("/Users/eric/Public/work/code/mne-project/ArwodMNE/out/Release/tests")
