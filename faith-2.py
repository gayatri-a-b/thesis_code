import numpy
import pandas
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# metrics: bank's profit
# how much disadvantaged group catches (ad - disad, earth mover)
# averages (to check for leveling down -- compare advanteaged group to itself)

# max util, eo, 

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
        average = (( 1*pin[0] + 2*pin[1] + 3*pin[2] + 4*pin[3] + 5*pin[4] + 6*pin[5] + 7*pin[6] ) / sum(pin))
        
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
                current_repayment = certainty_0_copy[decile] # current bin repayment certainty

                # if repaid
                if (repayment_truth == 1):
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
                current_repayment = certainty_1_copy[decile] # current bin repayment certainty

                # if repaid
                if (repayment_truth == 1):
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
        earth_mover_distance_initial = wasserstein_distance(self.initial_pi_0, self.initial_pi_1)
        earth_mover_distance_after = wasserstein_distance(self.pi_0, self.pi_1)

        # level-ing down
        change_average_pi_1 = self.average_score(self.pi_1) - self.average_score(self.initial_pi_1)

        # updated 
        return (updated_pi_0, updated_pi_1, bank_profit_iterated, earth_mover_distance_initial, earth_mover_distance_after, change_average_pi_1)



## TEST ITERATIONS
# test iterations
iterations = 2000

OS = OneStep()
(updated_pi_0_gb, updated_pi_1_gb, bank_profit_iterated_gb, earth_mover_distance_initial_gb, earth_mover_distance_after_gb, change_average_pi_1_gb) = OS.iterate(iterations, OS.gb_one_step)
(updated_pi_0_max_util, updated_pi_1_max_util, bank_profit_iterated_max_util, earth_mover_distance_initial_max_util, earth_mover_distance_after_max_util, change_average_pi_1_max_util) = OS.iterate(iterations, OS.max_one_step)

'''
updated_pi_0 = updated_pi_0_gb
updated_pi_1 = updated_pi_1_gb
folder_0 = 'gb-charts-iterations-group-0/'
folder_1 = 'gb-charts-iterations-group-1/'
'''

updated_pi_0 = updated_pi_0_max_util
updated_pi_1 = updated_pi_1_max_util
folder_0 = 'max-charts-iterations-group-0/'
folder_1 = 'max-charts-iterations-group-1/'


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