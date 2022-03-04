import gym
import simple_driving
import time
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from simple_driving.resources.wrapper import ProcessFrame84
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

def main():
    log_dir = "./gym/"
    os.makedirs(log_dir,exist_ok = True)
    #Create environment
    env = gym.make('SimpleDriving-v0')
    #Wrapper to convert oservation space from coordinate to RGBimage
    env = ProcessFrame84(env)
    env = Monitor(env,log_dir)
    #Convert env to vector and normalize it
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    model = PPO('CnnPolicy', env, verbose=1,learning_rate = 0.00025,tensorboard_log="./Simple-driving/",n_steps=10000,batch_size=1000,gamma=0.9995)
    
    model.learn(total_timesteps=150000)
    model.save("PPO_car")
    # model.load("PPO_car")
    
    print("************************************Start test************************************")
    ob = env.reset()
    while True:
        action, _state = model.predict(ob, deterministic=True)
        ob, reward, done, _ = env.step(action)
        print("reward",reward)
        env.render()
        if done:
            ob = env.reset()
            time.sleep(1/30)

if __name__ == '__main__':
    main()

#Run tensorboard with
    # tensorboard --logdir ./Simple-driving/