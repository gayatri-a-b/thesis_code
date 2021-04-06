# Senior Thesis Code
## Balasubramanian

This works off *"Fairness Is Not Static: Deeper Understanding of Long Term Fairness via Simulation Studies."* The code simulated the bank lending environment described in *"Delayed Impact of Fair Machine Learning."*

There are four agents:
- **GB**: loans if expectated outcome of loan decision does not decrease group's average credit score and bank cash does not dip before its minimum amount
-  **Max Utility**: loans if expected outcome of loan decisions does not decrease total bank cash
-  **EO with minimum limit**: calculates thresholds that maximize EO quantity and bank cash and loans at those thesholds; imposes a minimum limit on group 0's threshold such that it is 0.5% repayment certainty
 - **EO with NO minimum limit**: calculates thresholds that maximize EO quantity and bank cash and loans at those thesholds; no limit on what group 0's threshold must be

The user can dabble with the tests beginning on line 713 in order to run and understand the code.

The two cases studied in the paper are given in the tests provided in that section:
```python
# example case
pi_0 = [10, 10, 20, 30, 30, 0, 0]
pi_1 = [0, 10, 10, 20, 30, 30, 0]

# for active harm
pi_0_test = [60, 10, 0, 0, 0, 0, 30]
pi_1_test  = [0, 0, 40, 10, 0, 0, 50]
```

The agents can be run like this respectively:
```python
OS_gb = OneStep()
OS_gb = OneStep(pi_0 = pi_0_test, pi_1 = pi_1_test, bank_cash = bank_cash_test)
(updated_pi_0_gb, updated_pi_1_gb, bank_profit_iterated_gb, earth_mover_distance_initial_gb, earth_mover_distance_after_gb, earth_mover_distance_0_gb, earth_mover_distance_1_gb, change_average_pi_0_gb, change_average_pi_1_gb, total_loans_0_gb, total_loans_1_gb, successful_loans_0_gb, successful_loans_1_gb, successful_loans_total_0_gb, successful_loans_total_1_gb) = OS_gb.iterate(iterations, OS_gb.gb_one_step, False) # always pass in False

OS_max_util = OneStep(pi_0 = pi_0_test, pi_1 = pi_1_test, bank_cash = bank_cash_test)
(updated_pi_0_max_util, updated_pi_1_max_util, bank_profit_iterated_max_util, earth_mover_distance_initial_max_util, earth_mover_distance_after_max_util, earth_mover_distance_0_max_util, earth_mover_distance_1_max_util, change_average_pi_0_max_util, change_average_pi_1_max_util, total_loans_0_max_util, total_loans_1_max_util, successful_loans_0_max_util, successful_loans_1_max_util, successful_loans_total_0_max_util, successful_loans_total_1_max_util) = OS_max_util.iterate(iterations, OS_max_util.max_one_step, False) # always pass in False

OS_eo = OneStep()
OS_eo = OneStep(pi_0 = pi_0_test, pi_1 = pi_1_test, bank_cash = bank_cash_test)
(updated_pi_0_eo, updated_pi_1_eo, bank_profit_iterated_eo, earth_mover_distance_initial_eo, earth_mover_distance_after_eo, earth_mover_distance_0_eo, earth_mover_distance_1_eo, change_average_pi_0_eo, change_average_pi_1_eo, total_loans_0_eo, total_loans_1_eo, successful_loans_0_eo, successful_loans_1_eo, successful_loans_total_0_eo, successful_loans_total_1_eo) = OS_eo.iterate(iterations, OS_eo.eo_one_step, False) # always pass in False to run EO with limit
    
OS_eo_no_limit = OneStep(pi_0 = pi_0_test, pi_1 = pi_1_test, bank_cash = bank_cash_test)
(updated_pi_0_eo_no_limit, updated_pi_1_eo_no_limit, bank_profit_iterated_eo_no_limit, earth_mover_distance_initial_eo_no_limit, earth_mover_distance_after_eo_no_limit, earth_mover_distance_0_eo_no_limit, earth_mover_distance_1_eo_no_limit, change_average_pi_0_eo_no_limit, change_average_pi_1_eo_no_limit, total_loans_0_eo_no_limit, total_loans_1_eo_no_limit, successful_loans_0_eo_no_limit, successful_loans_1_eo_no_limit, successful_loans_total_0_eo_no_limit, successful_loans_total_1_eo_no_limit) = OS_eo_no_limit.iterate(iterations, OS_eo_no_limit.eo_one_step, True) # always pass in True to run EO with NO limit
```

Lastly, there are several variables that the user can pass into the ```OneStep``` class's constructor:
```python
pi_0 = [10, 10, 20, 30, 30, 0, 0], # starting distribution for group 0 (disadvantaged group)
pi_1 = [0, 10, 10, 20, 30, 30, 0], # starting distribution for group 1 (advantaged group)
certainty_0 = [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7], # repayment certainty for group 0 (disadvantaged group),
certainty_1 = [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7], # repayment certainty for group 1 (advantaged group), same for both groups in the model
group_chance = 0.5, # change of picking each group
loan_amount = 10, # amount requested on every loan
interest_rate = 2, # interest rate on every loan
bank_cash = 10000 # bank starting cash
bank_minimum = 10000 # minimum amount that bank should have
```
