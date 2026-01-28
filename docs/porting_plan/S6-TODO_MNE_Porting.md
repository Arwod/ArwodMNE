# TODO List for MNE Porting

## Environment Setup
- [ ] **Python Environment**: Install `numpy`, `scipy` in your python environment to run the verification script.
  ```bash
  pip install numpy scipy
  ```
- [ ] **Test Data Generation**: Run the generation script once environment is ready:
  ```bash
  cd ArwodMNE
  python3 tests/verification_data/gen_test_data.py
  ```
- [ ] **C++ Build**: Ensure `ArwodMNE` compiles with Qt 5/6 and Eigen.

## Configuration
- [ ] **Data Path**: Set `MNE_DATA` environment variable to share data between Python and C++.
