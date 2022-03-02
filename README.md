# Self-driving car with RL stable baseline3
Most of the project develop from 
>https://github.com/GerardMaggiolino/Gym-Medium-Post 
Please check it out!
This project focus on training self-driving car env by implementing PPO algorithm from stable baseline3

## Installation

Clone the project 

```bash
git clone 
```
Then run main.py

## Update
- Wrap env to change observation space from box to RGB image
- Using PPO with CNN policy instead of TRPO
- Normalize action space
- Add limited timestep reset condition
- Normalize distance in reward function

## Contributing
Sornsiri Promma

Thanks original project from Gerard Maggiolino

Please make sure to update tests as appropriate.
