# Self-driving car with RL stable baseline3
>Most of the project develop from https://github.com/GerardMaggiolino/Gym-Medium-Post 
Please check it out!

This project focus on training self-driving car env by implementing PPO algorithm from stable baseline3
<img src="https://github.com/SornsiriP/Self-Driving-car/blob/main/Example/Example.gif" width="80%">

## Installation

Clone the project 
```bash
git clone https://github.com/SornsiriP/Self-Driving-car
```
Then run Gym-Medium-Post/main.py

## Update
- Wrap env to change observation space from box to RGB image
  ```
  from simple_driving.resources.wrapper import ProcessFrame84
  
  env = ProcessFrame84(env)
  ```
- Using PPO with CNN policy instead of TRPO
  ```
  from stable_baselines3 import PPO
  
  model = PPO('CnnPolicy', env, verbose=1,learning_rate = 0.00025,tensorboard_log="./Simple-driving/",n_steps=10000,batch_size=1000,gamma=0.9995)
  model.learn(total_timesteps=150000)
  ```
- Normalize action space
  ```
  def map_action(self, action):
    speed_range = [0,1]
    steer_range = [-0.6,0.6]
    new_speed = np.interp(action[0],[-1,1],speed_range)
    new_steer = np.interp(action[0],[-1,1],steer_range)
    return [new_speed, new_steer]
  ```
- Add limited timestep reset condition
  ```
  if self.current_step >1000:
    self.current_step = 0
    self.done = True
  ```
- Normalize distance in reward function
  ```
  previous_dist_to_goal = np.linalg.norm(tuple(map(lambda i, j: i - j, self.goal, self.prev_pos)))
  current_dist_to_goal =  np.linalg.norm(tuple(map(lambda i, j: i - j, self.goal, car_ob[0:2])))
  ```

## Reference
https://github.com/GerardMaggiolino/Gym-Medium-Post 

https://www.etedal.net/2020/04/pybullet-panda_3.html

## Contributing
Sornsiri Promma


Thanks original project from Gerard Maggiolino

Please make sure to update tests as appropriate.
