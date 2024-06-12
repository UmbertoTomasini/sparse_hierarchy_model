Files to create SRHM data. These data are generated and then stored for later use (like training neural nets on them).

1. __init__.py
2. create_dataset-B.py for generating data with sparsity A/B
3. create_dataset-MOD.py for generating modified data with diffeo/synonyms
4. hierarchical_s0_bs_S contains the functions to generate data with sparsity A (called by 2)
5. hierarchical_s0_bs_S_diffeoB contains the functions to generate data with sparsity B (called by 2)
6. hierarchical_s0_bs_S_MOD contains the functions to generate data modified (called by 3)
7. script_data.py to create the bash files.sh
8. script_data-MOD.py to create the bash files.sh
9. utils_data.py
  
  
