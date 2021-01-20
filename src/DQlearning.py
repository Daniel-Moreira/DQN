import tensorflow as tf
import numpy as np
import retro

import os
import sys
import time

from PIL import Image
# from skimage import transform
# from skimage.color import rgb2gray

import matplotlib.pyplot as plt

from collections import deque

from Model import Model

import warnings
warnings.filterwarnings('ignore')

env = retro.make(game='SpaceInvaders-Atari2600', record='space_invaders/record')

STATE_SIZE = [110, 84, 4]
ACTION_SIZE = env.action_space.n
LEARNING_RATE =  0.00025
DISCOUNT_RATE = 0.99
BATCH_SIZE = 32
EXPLORE_PROB = [1.0, 0.1]
DECAY_RATE = 0.00001
SEND_TIME = 30*60 # 30 minutes
SKIP_FRAMES = 4
REPLAY_START_SIZE = 50000
# REPLAY_START_SIZE = 128

preprocessor_hyperparameters = {
    'CHOP_IMAGE': 12,
    'RESIZE': [84, 110],
    'STACK_SIZE': 4
}

print('The size of our frame is: ', env.observation_space)
print('The action size is: ', ACTION_SIZE)

# Here we create an hot encoded version of our actions
possible_actions = np.array(np.identity(ACTION_SIZE, dtype=int).tolist())
print('Set of actions: \n', possible_actions)
for i in range(8):
    print(env.get_action_meaning(possible_actions[i]))

# Simplify input
def preprocessor_image(image):
    chop_image = preprocessor_hyperparameters['CHOP_IMAGE']
    resize_image = preprocessor_hyperparameters['RESIZE']
    
    # Chop some pixels from image
    if chop_image > 0:
        image = image[8:-chop_image, 4:-chop_image]

    # Grayscale image
    # Downsize the image with anti-aliasing
    img = Image.fromarray(image, mode='RGB').convert('L').resize(resize_image)

    # img.show()
    return np.array(img)/255.0

# def preprocessor_image(image):
#     chop_image = preprocessor_hyperparameters['CHOP_IMAGE']
#     resize_image = preprocessor_hyperparameters['RESIZE']
    
#     # Grayscale image
#     gray_image = rgb2gray(image)

#     # Chop some pixels from image
#     if chop_image > 0:
#         gray_image = gray_image[8:-chop_image, 4:-chop_image]

#     # Center the mean around 0
#     normalize_image = gray_image/255.0

#     # Downsize the image with anti-aliasing
#     return transform.resize(normalize_image, resize_image)

# Stack images to have a sense of motion 
def stack_images(stacked_images, state, restart = False):
    stack_size = preprocessor_hyperparameters['STACK_SIZE']

    image_preprocessed = preprocessor_image(state)

    # Start with the same image copied n times
    if restart: stacked_images = deque([image_preprocessed for i in range(stack_size)], maxlen=stack_size)
    # Append a new image and remove the pre
    else: stacked_images.append(image_preprocessed)

    # Build an X:Y:Z image state
    stacked_state = np.stack(stacked_images, axis=2)
    return stacked_state, stacked_images

# import csv

# Write some statistics
# def summary(model, episode, reward):
#     avg_reward, std_reward, min_reward, max_reward = run_episode(model, False, False, 10)

#     with open('space_invaders/record/DQ_statistics.csv', mode='a+') as DQ_statistics:
#         writer = csv.writer(DQ_statistics)
#         if os.stat('space_invaders/record/DQ_statistics.csv').st_size == 0:
#             writer.writerow(['Episode', 'Reward', 'Avg', 'Std', 'Min', 'Max'])

#         writer.writerow([episode, reward, avg_reward, std_reward, min_reward, max_reward])
    
#     print(f'EPISODE:  {episode}, REWARD: {reward}, AVG: {avg_reward}, STD: {std_reward}, MIN: {min_reward}, MAX: {max_reward}') 
    
def run_episode(model, train, render, episodes):
    step = 0
    starttime = time.time()
    uploadTimes = 0

    decay = 0
    explore_probability = 0.05

    for episode in range(episodes):
        if step >= episodes:
            break
        terminated = False
        image = env.reset() 

        current_state, stacked_images = stack_images(None, image, True)
        
        print(model.choose_action(current_state, 0, True))
        cumulative_reward = 0
        
        while not terminated:
            if step >= episodes:
                break

            if render:
                env.render()    
            
            if train:
                explore_probability = EXPLORE_PROB[1] + (EXPLORE_PROB[0] - EXPLORE_PROB[1]) * np.exp(-DECAY_RATE * step)

            action = model.choose_action(current_state, explore_probability)

            reward_s = 0
            for _ in range(SKIP_FRAMES):
                next_state, reward, terminated, _ = env.step(possible_actions[action])

                cumulative_reward += reward
                reward_s += reward
                if terminated:
                    break
            
            next_state, stacked_images = stack_images(stacked_images, next_state, False)

            if train:
                t = 1
                if terminated:
                    t = 0
                model.store_memory(current_state, possible_actions[action], reward_s/5, next_state, t)

            if train and REPLAY_START_SIZE <= step:
                model.evaluate_action()

            current_state = next_state
            step+=1

            if terminated:
                print(f'Episode: {episode}, Step: {step}, Reward: {cumulative_reward}, Explore: {explore_probability}')

# Execute simulations and train the model 
def training():
    model = Model(STATE_SIZE, ACTION_SIZE, LEARNING_RATE, DISCOUNT_RATE, BATCH_SIZE)

    render = False
    if len(sys.argv) > 1:
        render = sys.argv[1] == 'render'
    
    run_episode(model, False, render, 12500000)

training()