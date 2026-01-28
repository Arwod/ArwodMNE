import os
import subprocess
import glob

test_dir = "out/Release/tests"
test_binaries = glob.glob(os.path.join(test_dir, "test_*"))

failed_tests = []
passed_tests = []
crashed_tests = []

for test in test_binaries:
    if not os.access(test, os.X_OK):
        continue
    
    print(f"Running {os.path.basename(test)}...")
    try:
        result = subprocess.run([test], capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            passed_tests.append(test)
        else:
            print(f"FAILED: {test}")
            failed_tests.append((test, result.stdout + result.stderr))
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {test}")
        crashed_tests.append((test, "Timeout"))
    except Exception as e:
        print(f"CRASHED: {test}")
        crashed_tests.append((test, str(e)))

print("\n--- TEST SUMMARY ---")
print(f"Total: {len(test_binaries)}")
print(f"Passed: {len(passed_tests)}")
print(f"Failed: {len(failed_tests)}")
print(f"Crashed/Timeout: {len(crashed_tests)}")

if failed_tests:
    print("\n--- FAILED TEST DETAILS ---")
    for test, log in failed_tests:
        print(f"\nTest: {test}")
        # Print only the last few lines of log to avoid too much output
        lines = log.splitlines()
        for line in lines[-20:]:
            print(line)
