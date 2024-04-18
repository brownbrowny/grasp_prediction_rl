from gymnasium.envs.registration import register

register(
     id="GraspEnv-v0",
     entry_point="envs:GraspEnv",
     max_episode_steps=100,
)