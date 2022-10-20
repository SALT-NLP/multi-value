### MTurk_Setup

1. Go to "Manage" and click "Create New Qualification Type" for each of the following dialect-specific qualifications. Be sure to record the Qualification ID for each of these Qualifications

   1. **Friendly Name:** Dialect A
      **Description:** You speak Dialect A
   2. **Friendly Name:** Dialect B
      **Description:** You speak Dialect B
   3. **Friendly Name:** Dialect C
      **Description:** You speak Dialect C
   4. **Friendly Name:** Dialect D
      **Description:** You speak Dialect D
   5. **Friendly Name:** Dialect E
      **Description:** You speak Dialect E
      
      ...
      
      and so on up the alphabet, all the way to 
      
      **Friendly Name:** Dialect O
      **Description:** You speak Dialect O

2. In a text editor, change the contents of `qual_request.py` `aws_access_key_id` and  `aws_secret_access_key` to the correct values for your AWS / MTurk Requester account

   1. In a shell, run `python qual_request.py`
   2. Remember the output printed here as `QUAL_ID` for a later stage

3. Go to requester.mturk.com and under Create click New Project (use the "Other" template) 

4. Fill in the following details
   **Project Name:** VALUE-CoQA-Validation
   **Title:** Dialect Understanding
   **Description:** Your goal is to decide whether bits of text sound acceptable according to the grammar rules of your dialect.
   **Keywords:** linguistics, grammar, dialect
   **Reward per assignment:** $0.06 (assumes 30 seconds per task, which is proven reasonable)
   **Number of assignments per task:** 3

   **Time allotted per assignment:** 1 hour
   **Task expires in:** 20 days
   **Auto-approve and pay Workers in:** 5 days
   **Require that Workers be Masters to do your tasks:** No
   **Specify any additional qualifications Workers must meet to work on your tasks:**

   	* English Varieties Test has been granted
   	* Dialect B has been granted (starting with Indian English and later switch to Dialect C for Singapore English or Dialect D for Appalachian or Dialect E for Chicano)
   	* HIT Approval Rate (%) greater than or equal to 98
    * Number of HITs Approved greater than or equal to 500

5. Copy the contents of `HIT_general.html` into the "Design Layout" pane and finish

6. Publish batches: Pull batches from the corresponding folders:

   1. `HIT_input/A/all_features_25.csv`: Dialect A
   2. `HIT_input/B/all_features_25.csv`: Dialect B
      ...
   3. `HIT_input/O/all_features_25.csv`: Dialect O

7. Either periodically run the following or write a script to run this automatically every hour or so: `python assign_workers_dialect_qualifications.py --qual_id QUAL_ID --A A_QUAL_ID --B B_QUAL_ID --C C_QUAL_ID --D D_QUAL_ID --E E_QUAL_ID --F F_QUAL_ID --G G_QUAL_ID --H H_QUAL_ID --I I_QUAL_ID --J J_QUAL_ID --K K_QUAL_ID --L L_QUAL_ID --M M_QUAL_ID --N N_QUAL_ID --O O_QUAL_ID` 

   * This is to link the dialect qual survey response to the specific qualifications for each dialect we are looking at.
