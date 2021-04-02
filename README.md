# Senior Thesis Code
## Balasubramanian

This works off "Fairness Is Not Static: Deeper Understanding of Long Term Fairness via Simulation Studies." The code simulated the bank lending environment described in "Delayed Impact of Fair Machine Learning."

There are four agents:
- GB: loans if expectated outcome of loan decision does not decrease group's average credit score and bank cash does not dip before its minimum amount
-  Max Utility: loans if expected outcome of loan decisions does not decrease total bank cash
-  EO with minimum limit: calculates thresholds that maximize EO quantity and bank cash and loans at those thesholds; imposes a minimum limit on group 0's threshold such that it is 0.5% repayment certainty
 - EO with NO minimum limit: calculates thresholds that maximize EO quantity and bank cash and loans at those thesholds; no limit on what group 0's threshold must be

The user can dabble with the tests beginning on line 672 in order to run and understand the code.

The two cases studied in the paper are given in the tests provided in that section:
```python
# example case
pi_0 = [10, 10, 20, 30, 30, 0, 0]
pi_1 = [0, 10, 10, 20, 30, 30, 0]

# for active harm
#pi_0_test = [60, 10, 0, 0, 0, 0, 30]
#pi_1_test  = [0, 0, 40, 10, 0, 0, 50]
```
