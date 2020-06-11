eval $(pyenv init -)
pyenv shell 3.6.8
poetry run python run_env.py jt_nog ddpg
poetry run python run_env.py pd_nog ppo
poetry run python run_env.py jt_g ddpg
poetry run python run_env.py pd_nog ddpg
poetry run python run_env.py pd_g ddpg
poetry run python run_env.py jt_nog ppo
poetry run python run_env.py jt_g ppo
poetry run python run_env.py pd_g ppo
