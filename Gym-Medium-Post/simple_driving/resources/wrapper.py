import gym
import numpy as np
import cv2



class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.env = env
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 84, 84), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(self.env.render(mode='rgb_array'))

    def process(frame):
        #print(frame.shape)
        if frame.size == 100 * 100 * 3:
            img = np.reshape(frame, [100, 100, 3]).astype(
                np.float32)
        else:
            assert False, "Unknown resolution."
        # img = img[:, :, 0] * 0.399 + img[:, :, 1] * 0.587 + \
        #       img[:, :, 2] * 0.114 #to one channel

        #print(img.shape)
        resized_screen = cv2.resize(
            img, (112, 84), interpolation=cv2.INTER_AREA)
        
        y_t = resized_screen[:, 14:98]
        y_t = np.reshape(y_t, [84, 84, 3])
        y_t = np.moveaxis(y_t, 2, 0)

        return y_t.astype(np.uint8)

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        new_shape = (old_shape[-1], old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        #print(np.moveaxis(observation, 2, 0).shape)
        #observation = observation/255 #normalization
        return np.moveaxis(observation, 2, 0)

