import copy

import numpy

if __name__ == '__main__':

    r = 100.0
    rewards = [[r, -1, 10], [-1, -1, -1], [-1, -1, -1]]
    x = [[rewards[0][0], rewards[0][1], rewards[0][2]],
         [rewards[1][0], rewards[1][1], rewards[1][2]],
         [rewards[2][0], rewards[2][1], rewards[2][2]]]
    y = [[rewards[0][0], rewards[0][1], rewards[0][2]],
         [rewards[1][0], rewards[1][1], rewards[1][2]],
         [rewards[2][0], rewards[2][1], rewards[2][2]]]
    g = 0.99
    up = 'up'
    up_mask = (-1, 0)
    down = 'down'
    down_mask = (1, 0)
    right = 'right'
    right_mask = (0, 1)
    left = 'left'
    left_mask = (0, -1)
    stay = 'stay'
    stay_mask = (0, 0)
    action_options = [up, down, right, left]
    actions = {up: [(up_mask, 0.8), (left_mask, 0.1), (right_mask, 0.1)],
               down: [(down_mask, 0.8), (left_mask, 0.1), (right_mask, 0.1)],
               right: [(right_mask, 0.8), (up_mask, 0.1), (down_mask, 0.1)],
               left: [(left_mask, 0.8), (up_mask, 0.1), (down_mask, 0.1)],
               stay: [(stay_mask, 1.0)]}
    state_actions = [[action_options, action_options, [stay]],
                     [action_options, action_options, action_options],
                     [action_options, action_options, action_options]]


    def limitCheck(n, bottom, ceil):
        if n < bottom:
            return bottom
        if n > ceil:
            return ceil
        return n


    def getNextPossibleStatesAndProbabilities(s, a):
        result = []
        ceil = len(x[0]) - 1
        action_mask_probability_tuple_list = actions[a]
        for t in action_mask_probability_tuple_list:
            action_mask = t[0]
            next_state = tuple(numpy.add(s, action_mask))
            if not (0 <= next_state[0] < 3 and 0 <= next_state[1] < 3):
                next_state = (limitCheck(next_state[0], 0, ceil), next_state[1])
                next_state = (next_state[0], limitCheck(next_state[1], 0, ceil))
            probability = t[1]
            result.append((next_state, probability))
        return result


    def euclideanDistance(u, u_tag):
        return numpy.sqrt(numpy.sum(numpy.power(numpy.subtract(u_tag, u), 2)))

    distance = 100
    iter_num = 1
    while distance > 20:

        print(f"iteration number {iter_num}:")
        iter_num += 1
        for i in range(3):
            for j in range(3):
                current_state = (i, j)
                print(f"Calculation of the current utility U'({current_state})")
                possible_actions = state_actions[i][j]
                next_utilities = []

                next_utilities_tag_str = []
                next_utilities_calc_str = []
                for a in possible_actions:
                    sum = 0
                    utility_nominees_str = ""
                    calc_u_tag_str_sum = ""
                    value_calc_str_sum = ""
                    for t in getNextPossibleStatesAndProbabilities(current_state, a):
                        k = t[0][0]
                        m = t[0][1]
                        probability = t[1]
                        probability_str = f"P(({k},{m})|{current_state},{a})"
                        u_str = f"U(({k},{m}))"
                        if calc_u_tag_str_sum == "":
                            calc_u_tag_str_sum += f"{u_str} * {probability_str}"
                            value_calc_str_sum += f"{x[k][m]} * {probability}"
                        else:
                            calc_u_tag_str_sum += f" + {u_str} * {probability_str}"
                            value_calc_str_sum += f" + {x[k][m]} * {probability}"
                        sum += x[k][m] * probability
                    next_utilities.append((a, sum))
                    next_utilities_tag_str.append(calc_u_tag_str_sum)
                    next_utilities_calc_str.append(value_calc_str_sum)
                reward = rewards[i][j]
                actions_curr, util_vals = zip(*next_utilities)
                max_utility = max(util_vals)
                y[i][j] = reward + g * max_utility
                max_action = actions_curr[util_vals.index(max_utility)]
                print(
                    f"U\'(({i},{j})) = R(({i},{j})) + gamma*max({next_utilities_tag_str}) = {reward} + {g}*max({next_utilities_calc_str}) = {y[i][j]}")
                print(f"Policy for {current_state} is {max_action}")
        # print(x)
        # print(y)
        distance = euclideanDistance(x, y)
        print(distance)
        x = copy.deepcopy(y)
