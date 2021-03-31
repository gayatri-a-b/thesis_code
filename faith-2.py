import math
import numpy
import pandas
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# metrics: bank's profit
# how much disadvantaged group catches (ad - disad, earth mover)
# averages (to check for leveling down -- compare advanteaged group to itself)

# max util, eo (the ratio of successful loans granted to the number of applicants who could have repaid)

class OneStep:
##### constructor
    def __init__( self, 
                  pi_0 = [10, 10, 20, 30, 30, 0, 0],
                  pi_1 = [0, 10, 10, 20, 30, 30, 0],
                  certainty_0 = [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7],
                  certainty_1 = [0.1, 0.2, 0.45, 0.6, 0.65, 0.7, 0.7],
                  group_chance = 0.5,
                  loan_amount = 10,
                  interest_rate = 1, #1 in paper
                  bank_cash = 10000 
                ):
    
        # store variables
        self.pi_0 = pi_0
        self.certainty_0 = certainty_0

        self.pi_1 = pi_1
        self.certainty_1 = certainty_1

        self.group_chance = group_chance
        self.loan_amount = loan_amount
        self.interest_rate = interest_rate
        self.bank_cash = bank_cash
        self.bank_minimum = bank_cash

        self.initial_pi_0 = pi_0
        self.initial_pi_1 = pi_1
        self.initial_bank_cash = bank_cash

        self.total_loans_0 = 0
        self.total_loans_1 = 0
        self.succesful_repayments_0 = 0
        self.succesful_repayments_1 = 0

        self.total_individuals_0 = 0
        self.total_individuals_1 = 0


##### return an individual and their outcome
    def get_person(self):
        # group
        group = numpy.random.choice(2, 1, p=[1 - self.group_chance, self.group_chance])[0]

        # decile (credit score bin)
        decile = numpy.random.choice(7, 1, p=[1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])[0]
        
        # repayment or default outcome
        if group == 0:
            repayment = numpy.random.choice(2, 1, p=[1 - self.certainty_0[decile], self.certainty_0[decile]])[0]
        else:
            repayment = numpy.random.choice(2, 1, p=[1 - self.certainty_1[decile], self.certainty_1[decile]])[0]

        # loan by dault
        loan_decision = 1

        # don't loan if nobody in the bin
        if group == 0 and self.pi_0[decile] == 0 :
            loan_decision = 0 
        if group and self.pi_1[decile] == 0:
            loan_decision = 0

        # return individual's outcome
        return ((group, decile, repayment, loan_decision))


##### return average score of a distribution
    def average_score(self, pin):
        average = (( 1*pin[0] + 2*pin[1] + 3*pin[2] + 4*pin[3] + 5*pin[4] + 6*pin[5] + 7*pin[6] ) / numpy.sum(pin))
        
        # return average
        return average


##### return expected change in bank cash (above minimum) and average credit score -- gb agent
    def gb_expected_update(self, group, decile):
        # copies of variables
        pi_0_repaid = self.pi_0.copy()
        pi_0_default = self.pi_0.copy()
        certainty_0 = self.certainty_0.copy()

        pi_1_repaid = self.pi_1.copy()
        pi_1_default = self.pi_1.copy()
        certainty_1 = self.certainty_1.copy()

        bank_cash_current = self.bank_cash

        ## expected bank amount (above minimum)
        if group == 0:
            repayment_certainty_bc = certainty_0[decile] # bin repayment certainty
        else:
            repayment_certainty_bc = certainty_1[decile] # bin repayment certainty

        # expectation of bank money after loan
        bank_cash_update = bank_cash_current + repayment_certainty_bc * self.loan_amount * self.interest_rate + (1 - repayment_certainty_bc) * ( -1 * self.loan_amount ) - self.bank_minimum


        ## expected average credit score
        if group == 0:
            repayment_certainty_av = certainty_0[decile] # current bin repayment certainty
            current_average = self.average_score(pi_0_repaid) # current average credit score

            # repaid
            # if we can move up
            if decile != 6:
                pi_0_repaid[decile + 1] += 1
                pi_0_repaid[decile] -= 1

            repaid_av = (self.average_score(pi_0_repaid) - current_average)

            # default
            # if we can move down
            if decile != 0:
                pi_0_default[decile - 1] += 1
                pi_0_default[decile] -= 1

            default_av = (self.average_score(pi_0_default) - current_average)
        else:
            repayment_certainty_av = certainty_1[decile] # current bin repayment certainty
            current_average = self.average_score(pi_1_repaid) # current average credit score

            # repaid
            # if we can move up
            if decile != 6:
                pi_1_repaid[decile + 1] += 1
                pi_1_repaid[decile] -= 1

            repaid_av = (self.average_score(pi_1_repaid) - current_average)

            # default
            # if we can move down
            if decile != 0:
                pi_1_default[decile - 1] += 1
                pi_1_default[decile] -= 1

            default_av = (self.average_score(pi_1_default) - current_average)

        # expectation of average credit score (change)
        average_update = (repayment_certainty_av*repaid_av + (1- repayment_certainty_av)*default_av)


        # return expected bank profit (above minimum) and credit score
        return ((bank_cash_update, average_update))


##### return expected change in bank cash -- max util agent
    def max_util_expected_update(self, group, decile):
        # copies of variables
        certainty_0 = self.certainty_0.copy()
        certainty_1 = self.certainty_1.copy()

        ## expected bank amount (above minimum)
        if group == 0:
            repayment_certainty_bc = certainty_0[decile] # bin repayment certainty
        else:
            repayment_certainty_bc = certainty_1[decile] # bin repayment certainty

        # expectation of bank money after loan
        bank_cash_update = repayment_certainty_bc * self.loan_amount * self.interest_rate + (1 - repayment_certainty_bc) * ( -1 * self.loan_amount )

        # return expected bank profit
        return (bank_cash_update)


    # define the roc curve a group
    def helper_roc (self, group): 
        # get the group data
        if group == 0:
            pi = self.pi_0
            certainty = self.certainty_0
        elif group == 1:
            pi = self.pi_1
            certainty = self.certainty_1

        # major points
        roc = [1.0]
        
        # for each bin endpoint
        for i in range(1, 7):

            # probability success and selected
            numerator = 0

            # probability success
            denominator = 0

            # for each individual
            for decile, bin_count in enumerate(pi):
                
                # set bin selection
                if decile < i:
                    bin_selection = 0
                else:
                    bin_selection = 1
                
                numerator += bin_count*bin_selection*certainty[decile]
                denominator += bin_count*certainty[decile] / numpy.sum(pi)
            
            # add the major threshold point to the array
            roc.append(numerator / denominator)

        # append the last point
        roc.append(0.0)

        # roc curve
        return (roc)  


    # return tpr given group 0's roc and threshold
    def helper_tpr (self, roc_0, t_0):
        (decimal_0, whole_0) = math.modf(t_0)
        whole_0 = int(whole_0)

        if (whole_0 > 0 and whole_0 < 7):
            tpr = (roc_0[whole_0 + 1] - roc_0[whole_0])*(t_0 - whole_0) + roc_0[whole_0]
        elif (whole_0 == 0):
            tpr = 1.0
        elif (whole_0 == 7):
            tpr = 0.0

        # true positive rate
        return (tpr)


    # return threshold for group 1 given group 1's roc and the tpr
    def helper_threshold (self, roc_1, tpr):
        if ((tpr < roc_1[0]) and (tpr > roc_1[1])):
            t_1 = ((tpr - roc_1[0]) / (roc_1[1] - roc_1[0])) + 0
        elif ((tpr < roc_1[1]) and (tpr > roc_1[2])):
            t_1 = ((tpr - roc_1[1]) / (roc_1[2] - roc_1[1])) + 1
        elif ((tpr < roc_1[2]) and (tpr > roc_1[3])):
            t_1 = ((tpr - roc_1[2]) / (roc_1[3] - roc_1[2])) + 2
        elif ((tpr < roc_1[3]) and (tpr > roc_1[4])):
            t_1 = ((tpr - roc_1[3]) / (roc_1[4] - roc_1[3])) + 3
        elif ((tpr < roc_1[4]) and (tpr > roc_1[5])):
            t_1 = ((tpr - roc_1[4]) / (roc_1[5] - roc_1[4])) + 4
        elif ((tpr < roc_1[5]) and (tpr > roc_1[6])):
            t_1 = ((tpr - roc_1[5]) / (roc_1[6] - roc_1[5])) + 5
        elif ((tpr < roc_1[6]) and (tpr > roc_1[7])):
            t_1 = ((tpr - roc_1[6]) / (roc_1[7] - roc_1[6])) + 6
        else:
            t_1 = 7.0
        
        # threshold
        return (t_1)


    # return the negative of the number to optimize for eo_expected_update
    def helper_function_to_optimize(self, t_0):
        # get the roc curve's major points for group 0 and group 1
        roc_0 = self.helper_roc(0)
        roc_1 = self.helper_roc(1)

        # get the tpr that both group 0 and group 1 share
        tpr = self.helper_tpr(roc_0, t_0)

        # get the threshold for group 1 given a tpr
        t_1 = self.helper_threshold (roc_1, tpr)

        # some local variables
        (decimal_0, whole_0) = math.modf(t_0)
        (decimal_1, whole_1) = math.modf(t_1)

        numerator_0 = 0
        numerator_1 = 0

        # for each individual in group 0
        for decile, bin_count in enumerate(self.pi_0):
            # set bin selection
            if decile < whole_0:
                bin_selection = 0
            elif decile == whole_0:
                bin_selection = 1 - decimal_0
            elif decile > whole_0:
                bin_selection = 1

            numerator_0 += bin_selection*(self.certainty_0[decile]*self.interest_rate*self.loan_amount + (1 - self.certainty_0[decile])*(-1 * self.loan_amount))
        
        # for each individual in group 1
        for decile, bin_count in enumerate(self.pi_1):
            # set bin selection
            if decile < whole_1:
                bin_selection = 0
            elif decile == whole_1:
                bin_selection = 1 - decimal_1
            elif decile > whole_1:
                bin_selection = 1

            numerator_1 += bin_selection*(self.certainty_1[decile]*self.interest_rate*self.loan_amount + (1 - self.certainty_1[decile])*(-1 * self.loan_amount))
        
        # final number to optimize
        number = (numerator_0 + numerator_1) / (numpy.sum(self.pi_0) + numpy.sum(self.pi_1))

        # return the negative of the number to optimize
        return (-1 * number)


    # optimizer for eo_expected_update to determine best thresholds for group 0 and group 1
    def eo_optimize(self):
        # limit threshold to repayment probability > 0.5
        index = next(x for x, val in enumerate(self.certainty_0) if val >= 0.5)
        upper_bin = self.certainty_0[index]
        lower_bin = self.certainty_0[index - 1]
        distance_from_lower_bin = 0.5 - lower_bin
        distance_from_upper_bin = upper_bin - 0.5
        t_0_lower_limit = self.certainty_0[index] - (distance_from_lower_bin / (distance_from_lower_bin + distance_from_upper_bin))

        # run the threshold optimization
        res = optimize.minimize(self.helper_function_to_optimize, 4.5, bounds=[(t_0_lower_limit,7)])
        t_0 = res.x
        
        # get the roc curve's major points for group 0 and group 1
        roc_0 = self.helper_roc(0)
        roc_1 = self.helper_roc(1)

        # get the tpr that both group 0 and group 1 share
        tpr = self.helper_tpr(roc_0, t_0)

        # get the threshold for group 1 given a tpr
        t_1 = self.helper_threshold (roc_1, tpr)

        # return the two thresholds
        return (t_0, t_1)


##### return actual update
    def actual_update(self, group, decile, loan_decision, repayment_truth):
        # current variable copies
        pi_0_copy = self.pi_0.copy()
        certainty_0_copy  = self.certainty_0.copy()

        pi_1_copy = self.pi_1.copy()
        certainty_1_copy  = self.certainty_1.copy()

        bank_cash_current = self.bank_cash

        # if we loan
        if (loan_decision == 1):

            if group == 0:
                # increment that we loaned
                self.total_loans_0 += 1
                
                current_repayment = certainty_0_copy[decile] # current bin repayment certainty

                # if repaid
                if (repayment_truth == 1):
                    # increment that we loaned and they repaid successful
                    self.succesful_repayments_0 += 1

                    # move up if possible
                    if decile != 6:
                        pi_0_copy[decile + 1] += 1
                        pi_0_copy[decile] -= 1

                        bank_cash_current += (self.interest_rate*self.loan_amount)
                else:
                    # move down if possible
                    if decile != 0:
                        pi_0_copy[decile - 1] += 1
                        pi_0_copy[decile] -= 1

                        bank_cash_current -= self.loan_amount
            else:
                # increment that we loaned
                self.total_loans_1 += 1

                current_repayment = certainty_1_copy[decile] # current bin repayment certainty

                # if repaid
                if (repayment_truth == 1):
                    # increment that we loaned and they repaid successful
                    self.succesful_repayments_1 += 1

                    # move up if possible
                    if decile != 6:
                        pi_1_copy[decile + 1] += 1
                        pi_1_copy[decile] -= 1

                        bank_cash_current += (self.interest_rate*self.loan_amount)
                else:
                    # move down if possible
                    if decile != 0:
                        pi_1_copy[decile - 1] += 1
                        pi_1_copy[decile] -= 1

                        bank_cash_current -= self.loan_amount

        # return updated distributions
        return ((pi_0_copy, pi_1_copy, bank_cash_current))


##### take one step -- gb agent
    def gb_one_step(self):
        # get person
        (group, decile, repayment_truth, loan) = self.get_person()

        # increment which group individual is from
        if group == 0:
            self.total_individuals_0 += 1
        else:
            self.total_individuals_1 += 1

        # if person exists
        if (loan == 1):
            # calculate expected update
            (bank_cash_update, average_update) = self.gb_expected_update(group, decile)

            if ((bank_cash_update >= 0) and (average_update >= 0)):
                loan_decision = 1
            elif (decile == 6):
                loan_decision = 1
            else:
                loan_decision = 0

            # if loaned then update:
            if (loan_decision == 1):
                (pi_0_copy, pi_1_copy, bank_cash_current) = self.actual_update(group, decile, loan_decision, repayment_truth)

                # update
                self.pi_0 = pi_0_copy
                self.pi_1 = pi_1_copy
                self.bank_cash = bank_cash_current


##### take one step -- max util agent
    def max_one_step(self):
        # get person
        (group, decile, repayment_truth, loan) = self.get_person()

        # increment which group individual is from
        if group == 0:
            self.total_individuals_0 += 1
        else:
            self.total_individuals_1 += 1

        # if person exists
        if (loan == 1):
            # calculate expected update
            bank_cash_update = self.max_util_expected_update(group, decile)

            if (bank_cash_update >= 0):
                loan_decision = 1
            else:
                loan_decision = 0

            # if loaned then update:
            if (loan_decision == 1):
                (pi_0_copy, pi_1_copy, bank_cash_current) = self.actual_update(group, decile, loan_decision, repayment_truth)

                # update
                self.pi_0 = pi_0_copy
                self.pi_1 = pi_1_copy
                self.bank_cash = bank_cash_current


##### take one step -- eo agent
    def eo_one_step(self):
        # get person
        (group, decile, repayment_truth, loan) = self.get_person()

        # increment which group individual is from
        if group == 0:
            self.total_individuals_0 += 1

        else:
            self.total_individuals_1 += 1

        # if person exists
        if (loan == 1):
            # get the lending rule
            (t_0, t_1) = self.eo_optimize()

            (decimal_0, whole_0) = math.modf(t_0)
            (decimal_1, whole_1) = math.modf(t_1)

            # group 0 loan decision
            if group == 0:
                if (decile < int(whole_0)):
                    loan_decision = 0
                elif (decile == int(whole_0)):
                    loan_decision = numpy.random.choice(2, 1, p=[decimal_0, 1 - decimal_0])[0]
                elif (decile > int(whole_0)):
                    loan_decision = 1

            # group 1 loan decision
            if group == 1:
                if (decile < int(whole_1)):
                    loan_decision = 0
                elif (decile == int(whole_1)):
                    loan_decision = numpy.random.choice(2, 1, p=[decimal_1, 1 - decimal_1])[0]
                elif (decile > int(whole_1)):
                    loan_decision = 1

            # if loaned then update:
            if (loan_decision == 1):
                (pi_0_copy, pi_1_copy, bank_cash_current) = self.actual_update(group, decile, loan_decision, repayment_truth)

                # update
                self.pi_0 = pi_0_copy
                self.pi_1 = pi_1_copy
                self.bank_cash = bank_cash_current


##### update specific number of times
    def iterate(self, iterations, funct):
        updated_pi_0 = []
        updated_pi_1 = []
        updated_bank_cash = []

        # index
        i = 0

        # only count successful updates
        while (i < iterations):
            funct()

            updated_pi_0.append(self.pi_0)
            updated_pi_1.append(self.pi_1)
            updated_bank_cash.append(self.bank_cash)

####            #print(self.pi_0)

            i += 1

        # profit
        bank_profit_iterated = self.bank_cash - self.initial_bank_cash

        # disadvantaged group catch up
        earth_mover_distance_initial = wasserstein_distance(u_values=[1,2,3,4,5,6,7],
                                                            u_weights=self.initial_pi_0/numpy.sum(self.initial_pi_0),
                                                            v_values=[1,2,3,4,5,6,7],
                                                            v_weights=self.initial_pi_1/numpy.sum(self.initial_pi_1))
        earth_mover_distance_after = wasserstein_distance(u_values=[1,2,3,4,5,6,7],
                                                          u_weights=self.pi_0/numpy.sum(self.pi_0),
                                                          v_values=[1,2,3,4,5,6,7],
                                                          v_weights=self.pi_1/numpy.sum(self.pi_1))

        # level-ing down / active harm
        change_average_pi_0 = self.average_score(self.pi_0) - self.average_score(self.initial_pi_0)
        change_average_pi_1 = self.average_score(self.pi_1) - self.average_score(self.initial_pi_1)

        # total loans
        try:
            total_loans_0 = self.total_loans_0 / self.total_individuals_0
        except:
            total_loans_0 = 0.0
        try:
            total_loans_1 = self.total_loans_1 / self.total_individuals_1
        except:
            total_loans_1 = 0.0

        # fraction of successful loans
        try:
            successful_loans_0 = self.succesful_repayments_0 / self.total_loans_0
        except:
            successful_loans_0 = 0.0
        try:
            successful_loans_1 = self.succesful_repayments_1 / self.total_loans_1
        except:
            successful_loans_1 = 0.0

        # updated 
        return (updated_pi_0, updated_pi_1, bank_profit_iterated, earth_mover_distance_initial, earth_mover_distance_after, change_average_pi_0, change_average_pi_1, total_loans_0, total_loans_1, successful_loans_0, successful_loans_1)


print("")
print("TESTS-----------")


## TEST ITERATIONS
# test iterations
iterations = 100
print("")

OS_gb = OneStep()
(updated_pi_0_gb, updated_pi_1_gb, bank_profit_iterated_gb, earth_mover_distance_initial_gb, earth_mover_distance_after_gb, change_average_pi_0_gb, change_average_pi_1_gb, total_loans_0_gb, total_loans_1_gb, successful_loans_0_gb, successful_loans_1_gb) = OS_gb.iterate(iterations, OS_gb.gb_one_step)

OS_max_util = OneStep()
(updated_pi_0_max_util, updated_pi_1_max_util, bank_profit_iterated_max_util, earth_mover_distance_initial_max_util, earth_mover_distance_after_max_util, change_average_pi_0_max_util, change_average_pi_1_max_util, total_loans_0_max_util, total_loans_1_max_util, successful_loans_0_max_util, successful_loans_1_max_util) = OS_max_util.iterate(iterations, OS_max_util.max_one_step)

OS_eo = OneStep()
(updated_pi_0_eo, updated_pi_1_eo, bank_profit_iterated_eo, earth_mover_distance_initial_eo, earth_mover_distance_after_eo, change_average_pi_0_eo, change_average_pi_1_eo, total_loans_0_eo, total_loans_1_eo, successful_loans_0_eo, successful_loans_1_eo) = OS_eo.iterate(iterations, OS_eo.eo_one_step)

print('Iterations: ' + str(iterations))

print("GB Policy")
print('bank profit: ' + str(bank_profit_iterated_gb))
print('intial earth movers dist btwn pi_0 and pi_1: ' + str(earth_mover_distance_initial_gb))
print('final earth movers dist btwn updated pi_0 and updated pi_1: ' + str(earth_mover_distance_after_gb))
print('change in pi_0s average credit score: ' + str(change_average_pi_0_gb))
print('change in pi_1s average credit score: ' + str(change_average_pi_1_gb))
print('fraction of successful loans in group 0: ' + str(successful_loans_0_gb))
print('fraction of successful loans in group 1: ' + str(successful_loans_1_gb))
print('fraction of applicants from group 0: ' + str(total_loans_0_gb))
print('fraction of applicants from group 1: ' + str(total_loans_0_gb))

print("")

print("Max Util Policy")
print('bank profit: ' + str(bank_profit_iterated_max_util))
print('intial earth movers dist btwn pi_0 and pi_1: ' + str(earth_mover_distance_initial_max_util))
print('final earth movers dist btwn updated pi_0 and updated pi_1: ' + str(earth_mover_distance_after_max_util))
print('change in pi_0s average credit score: ' + str(change_average_pi_0_max_util))
print('change in pi_1s average credit score: ' + str(change_average_pi_1_max_util))
print('fraction of successful loans in group 0: ' + str(successful_loans_0_max_util))
print('fraction of successful loans in group 1: ' + str(successful_loans_1_max_util))
print('fraction of applicants from group 0: ' + str(total_loans_0_max_util))
print('fraction of applicants from group 1: ' + str(total_loans_0_max_util))

print("")

print("EO Policy")
print('bank profit: ' + str(bank_profit_iterated_eo))
print('intial earth movers dist btwn pi_0 and pi_1: ' + str(earth_mover_distance_initial_eo))
print('final earth movers dist btwn updated pi_0 and updated pi_1: ' + str(earth_mover_distance_after_eo))
print('change in pi_0s average credit score: ' + str(change_average_pi_0_eo))
print('change in pi_1s average credit score: ' + str(change_average_pi_1_eo))
print('fraction of successful loans in group 0: ' + str(successful_loans_0_eo))
print('fraction of successful loans in group 1: ' + str(successful_loans_1_eo))
print('fraction of applicants from group 0: ' + str(total_loans_0_eo))
print('fraction of applicants from group 1: ' + str(total_loans_0_eo))

'''
updated_pi_0 = updated_pi_0_gb
updated_pi_1 = updated_pi_1_gb
folder_0 = 'gb-charts-iterations-group-0/'
folder_1 = 'gb-charts-iterations-group-1/'


updated_pi_0 = updated_pi_0_max_util
updated_pi_1 = updated_pi_1_max_util
folder_0 = 'max-charts-iterations-group-0/'
folder_1 = 'max-charts-iterations-group-1/'


updated_pi_0 = updated_pi_0_eo
updated_pi_1 = updated_pi_1_eo
folder_0 = 'max-charts-iterations-group-0/'
folder_1 = 'max-charts-iterations-group-1/'

'''

'''
for ind, arr in enumerate(updated_pi_0):
    iterations_ = ind

    if ((ind % 50) == 0):
        objects = ('100', '200', '300', '400', '500', '600', '700')
        y_pos = numpy.arange(len(objects))
        outcome_ = arr

        plt.bar(y_pos, outcome_, align='center', alpha=0.5)

        plt.xticks(y_pos, objects)
        plt.ylabel('Number of People')
        plt.xlabel('Credit Score')
        plt.title('Group 0 ' + str(iterations_) + " Iterations")
        plt.savefig(folder_0 + str(iterations_) + '.png')
        plt.clf()


for ind, arr in enumerate(updated_pi_1):
    iterations_ = ind

    if ((ind % 50) == 0):
        objects = ('100', '200', '300', '400', '500', '600', '700')
        y_pos = numpy.arange(len(objects))
        outcome_ = arr

        plt.bar(y_pos, outcome_, align='center', alpha=0.5)

        plt.xticks(y_pos, objects)
        plt.ylabel('Number of People')
        plt.xlabel('Credit Score')
        plt.title('Group 1 ' + str(iterations_) + " Iterations")
        plt.savefig(folder_1 + str(iterations_) + '.png')
        plt.clf()
'''

# (the ratio of successful loans granted to the number of applicants who could have repaid)
# if I set my threshold to a real value for group 0, want to compute what threshold for second group giving same tpr
# integer thresholds: sklearn.metrics.roc_curve
# linear within each bin so we can tpr for every real number
# for group 2, find the point that makes tpr

# fix threshold for group 1 --> lets me compute threshold for group 2
# now I can compute expected profit to lender: 
# now we can optimize
# https://github.com/google/ml-fairness-gym/blob/5b1cd336b844059aa4e4426b54d1f0e6b8c4c7e9/agents/threshold_policies.py#L40
# https://excalidraw.com/#room=9be8b216fdef5982fe6d,aMz4n7E-FUv6Of82e3Nkfg
# https://docs.scipy.org/doc/scipy/reference/optimize.html
