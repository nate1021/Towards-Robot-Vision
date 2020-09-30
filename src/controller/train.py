import sys
sys.path.append(__file__ + '/../..')
import tensorflow as tf
import numpy as np
import math
import sys
import time
from stopwatch import Stopwatch

import common
import controller.network
import controller.network_linker
import data_collection.camera_meta
import controller.draw_path


POPULATION_STD = 0.2

CONTROLLER_TIMESTEPS = 50  # larger encorages turning all the way before moving
BOARD_SPAWN_PERCENTAGE = 1  # 1 = 100% of 0.8


best_controller_path = '../../saved_models/best_controller.csv'


trackers_arr = data_collection.camera_meta.load_trackers()
print('%d positions ready.' % len(trackers_arr))

population = np.random.normal(loc=0, scale=POPULATION_STD, size=[controller.network.CONTROLLER_N, controller.network.gene_count])  # random speeds

print('Set test_mode to True to test the current controller''s performance.')
test_mode = False
if test_mode:
    population[0] = np.loadtxt(best_controller_path)
    print('Loaded best controller from: %s' % best_controller_path)

global_position_placeholder = tf.placeholder(name='pa', shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N, 2], dtype=tf.float32)  # global x, global y
global_orientation_placeholder = tf.placeholder(name='pb', shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N], dtype=tf.float32)  # global bearing

global_position = global_position_placeholder
global_orientation = global_orientation_placeholder
state = controller.network.first_hidden_state_tensor()

saved_positions = [global_position]
saved_orientations = [global_orientation]
with tf.variable_scope("", reuse=tf.AUTO_REUSE):

    print('Building giant graph:')
    for t in range(CONTROLLER_TIMESTEPS - 1):

        if t < 10:
            noise = 0
        else:
            shp = [controller.network.CONTROLLER_N, controller.network.LOCATION_N, 1]
            logic = tf.where(tf.random_uniform(shape=shp,minval=0,maxval=1) > 0.90, tf.ones(shp), tf.zeros(shp))
            noise = (3.0/5.0) * logic * tf.random_normal(shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N, common.LATENT_COUNT],mean=0,stddev=1.0)

        global_position, global_orientation, state, _, _ = controller.network_linker.next_position_orientation(global_position, global_orientation, state, noise)

        saved_positions.append(global_position)
        saved_orientations.append(global_orientation)
        print('Controller steps built: %d/%d' % (t + 2, CONTROLLER_TIMESTEPS))

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    controller.network_linker.restore_models(trackers_arr, sess)

    generation_i = 0
    stopwatch = Stopwatch()
    while generation_i < 2700:

        generations_per_save = 300
        for generations_per_save_countup in range(generations_per_save):
            save_generation = (generations_per_save_countup + 1 == generations_per_save) or generation_i == 0

            # calculate fitness
            gene_feed_dict = controller.network.gen_feed_dict(population)
            fitness_sum = np.zeros(shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N])  # higher is better
            for position_index in range(len(trackers_arr)):

                if save_generation:
                    controller.draw_path.clear(controller.network.LOCATION_N, False, position_index)
                    controller.draw_path.draw_trackers(trackers_arr[position_index])

                global_position_start = np.zeros(shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N, 2])  # global x, global y
                global_orientation_start = np.zeros(shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N])  # global bearing
                global_position_start[:] = np.random.uniform(low=-common.BOARD_RANGE * BOARD_SPAWN_PERCENTAGE, high=common.BOARD_RANGE * BOARD_SPAWN_PERCENTAGE,
                                                             size=[controller.network.LOCATION_N, 2])  # random positions
                global_orientation_start[:] = np.random.uniform(low=0, high=360, size=[controller.network.LOCATION_N])  # random bearing

                gene_feed_dict[global_position_placeholder] = global_position_start
                gene_feed_dict[global_orientation_placeholder] = global_orientation_start

                controller.network_linker.load_latent_simulator(position_index, sess)
                saved_positions_values, saved_orientations_values = sess.run((saved_positions, saved_orientations), feed_dict=gene_feed_dict)

                if save_generation:
                    controller.draw_path.draw_start_circles(saved_positions_values[0][0], saved_orientations_values[0][0], None)

                # mod 7 FITNESS function
                distance_from_target = np.zeros(shape=[CONTROLLER_TIMESTEPS, controller.network.CONTROLLER_N, controller.network.LOCATION_N])
                distance_over_boarder = np.zeros(shape=[CONTROLLER_TIMESTEPS, controller.network.CONTROLLER_N, controller.network.LOCATION_N])
                distance_from_anti_target = np.zeros(shape=[CONTROLLER_TIMESTEPS, controller.network.CONTROLLER_N, controller.network.LOCATION_N])

                pos_shp = [controller.network.CONTROLLER_N, controller.network.LOCATION_N, 2]
                for t in range(CONTROLLER_TIMESTEPS):
                    distance_from_target[t, :, :] = np.linalg.norm(saved_positions_values[t] - trackers_arr[position_index][common.TARGET_STRING], axis=2)
                    distance_from_anti_target[t, :, :] = np.linalg.norm(saved_positions_values[t] - trackers_arr[position_index][common.ANTITARGET_STRING], axis=2)
                    distance_over_boarder[t, :, :] = np.sum(np.maximum(np.zeros(pos_shp),
                                                                       np.abs(saved_positions_values[t]) - (common.BOARD_RANGE * np.ones(pos_shp))),  axis=2)

                still_traveling_to_target = np.ones(shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N], dtype=np.bool)  # true
                still_traveling_to_anti_target = np.ones(shape=[controller.network.CONTROLLER_N, controller.network.LOCATION_N], dtype=np.bool)  # true
                for t in range(1, CONTROLLER_TIMESTEPS):
                    currently_not_in_target = distance_from_target[t - 1] > common.TARGET_DESIRED_DISTANCE_TRAINING
                    still_traveling_to_target = np.logical_and(still_traveling_to_target,
                                                               currently_not_in_target)

                    currently_in_anti_target = distance_from_anti_target[t - 1] <= common.ANTI_TARGET_MIN_DISTANCE

                    distance_towards_target = np.where(still_traveling_to_target, distance_from_target[t - 1] - distance_from_target[t], 1)
                    distance_towards_antitarget = np.where(currently_in_anti_target, distance_from_anti_target[t - 1] - distance_from_anti_target[t], 0)
                    fitness_sum += distance_towards_target
                    fitness_sum -= np.maximum(0, 100 * distance_towards_antitarget)
                    fitness_sum -= distance_over_boarder[t] * 100

                # draw lines
                if save_generation:
                    for t in range(1, CONTROLLER_TIMESTEPS):
                        controller.draw_path.draw_lines(saved_positions_values[t][0], saved_positions_values[t - 1][0],
                                                        saved_orientations_values[t][0], saved_orientations_values[t - 1][0], (255, 0, 255))
                    controller.draw_path.save()
            fitness_sum = np.sum(fitness_sum, axis=1)

            # selection
            children_needed = int(math.ceil(1 + math.sqrt(1 + (8 * (controller.network.CONTROLLER_N / 2)))) / 2)
            arena_size = int(math.floor(controller.network.CONTROLLER_N / children_needed))

            global_best_i = 0
            best_parents = []
            for ci in range(0, children_needed):
                start_index = ci * arena_size
                if ci < children_needed - 1:
                    end_index = (ci + 1) * arena_size
                else:
                    end_index = controller.network.CONTROLLER_N

                best_parent_i = start_index
                for i in range(start_index + 1, end_index):
                    if fitness_sum[i] > fitness_sum[best_parent_i]:
                        best_parent_i = i

                if fitness_sum[best_parent_i] > fitness_sum[global_best_i]:
                    global_best_i = best_parent_i

                best_parents.append(population[best_parent_i])

            global_best = population[global_best_i]
            print('Generation %d, Average fitness: %f, Global best fitness: %f. Index: %d' %
                  (generation_i, np.average(fitness_sum), fitness_sum[global_best_i], global_best_i))
            if test_mode:
                sys.exit(0)

            # crossover
            new_population = []
            for i in range(children_needed):
                for j in range(i + 1, children_needed):
                    if len(new_population) == controller.network.CONTROLLER_N:
                        break

                    dad = best_parents[i]
                    mom = best_parents[j]

                    # simulatred binary crossover,
                    # according to: https://stackoverflow.com/questions/7280486/what-is-the-best-way-to-perform-vector-crossover-in-genetic-algorithm
                    average = (mom + dad) / 2
                    BETA = 0.8
                    difference = mom - dad
                    child1 = average - (0.5 * BETA * difference)
                    child2 = average + (0.5 * BETA * difference)

                    new_population.append(child1)
                    if len(new_population) == controller.network.CONTROLLER_N:
                        break
                    new_population.append(child2)

            # mutation
            twenty_p_on = np.random.uniform(0.0, 1.0, size=population.shape) >= 0.8
            mutants = np.where(twenty_p_on, np.random.normal(loc=0, scale=POPULATION_STD * 0.1, size=population.shape), np.zeros_like(population))
            new_population += mutants

            np.random.shuffle(new_population)
            population = new_population
            generation_i += 1

        print('Saving global best to: %s' % best_controller_path)
        np.savetxt(best_controller_path, global_best)
        print('Global best saved.')
        print('Trained for %d mins' % int(stopwatch.duration / 60.0))
