import os
import sys
import time
import logging

sys.path.append('./nao_search')

import gym

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy

from nao_search import epd
from nao_search.common.utils import random_seqences
from nao_search.common.utils import min_max_normalization
from nao_search.common.utils import pairwise_accuracy

from nao_search.common.logger import Logger
from nao_search.common.logger import LoggingConfig

from env_wrapper import ActionRemapWrapper, SkillWrapper


LoggingConfig.Use(filename='nao_skill_search_atari_(1, 9).train.log', 
                  output_to_file=True,
                  level='DEBUG')

LOG = Logger('nao_search')



# === Parameters ===
SKILL_LENGTH = 9
NUM_SKILLS_PER_SET = 1
ACTION_SPACE = 6

NUM_ITERATION = 5
UPDATE_BATCHSIZE = 128
CPU = 1
EPOCHS = 300
EVAL_INTERVAL = 50
LOG_INTERVAL = 1
TRAINING_STEPS = 5000000

ENV_ID = 'Alien-ramDeterministic-v4'
MODEL_SAVE_PATH = './models'
SKILL_SAVE_PATH = './skills'
TENSORBOARD_LOGDIR = './epd_logs'
TENSORBOARD_LOGNAME = 'nao_skill_search_atari_(1, 9)'
# ==================


# === Utility ===
def get_scores(skills):
    _scores = []

    for skill in skills:
        env_creator = lambda: ActionRemapWrapper(gym.make(ENV_ID))
        atari_manager = AtariPolicyManager(env_creator=env_creator, 
                                           model=PPO2, 
                                           policy=MlpPolicy,
                                           save_path='alien',
                                           verbose=0,
                                           num_cpu=15)
        ave_score, ave_action_reward = atari_manager.get_rewards(skill, train_total_timesteps=TRAINING_STEPS)

        _scores.append(ave_score)

    return _scores
    


def make_dirs(path):
    import errno
    try:
        os.makedirs(path)
    except e:
        if e.errno != errno.EEXIST:
            raise

def divide_skills(skills):
    def divide_skill(skill):
        
        assert len(skill) // NUM_SKILLS == SKILL_LENGTH

        avg = SKILL_LENGTH
        out = []
        last = 0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out

    divided_skills = []

    for skill in skills:
        divided_skills.append(divide_skill(skill))

    return divided_skills



def log_top_n_scores(N, _skills, _scores):
    _, top_n_scores = get_top_n(N=N,
                                seqs=_skills,
                                scores=_scores)

    for index, score in enumerate(top_n_scores):
        LOG.add_pair('Top {}'.format(index+1), score)


def output_skills(path, _skills, _scores):
    with open(path, 'w') as f:
        for skill, score in zip(_skills, _scores):
            f.write('{}:{}'.format(skill, scores))
# ===============


# === Initialize ===
make_dirs(MODEL_SAVE_PATH)
make_dirs(SKILL_SAVE_PATH)
TOTAL_SKILL_LENGTH = SKILL_LENGTH * NUM_SKILLS_PER_SET
# ==================


# === Generate random skills ===
skills = random_sequences(
                        length=SKILL_LENGTH,
                        seq_num=300,
                        vocab_size=ACTION_SPACE)

div_skills = divide_skills(skills)

scores = get_scores(div_skills)
# ==============================



# === Print top 10 ranking ===
LOG.set_header('Top 10 ranking')
log_top_n_scores(10, skills, scores)
LOG.dump_to_log(level=logging.INFO)
# ============================



# === Create model ===
model = epd.Model(
                  source_length=SKILL_LENGTH,
                  encoder_length=SKILL_LENGTH,
                  decoder_length=SKILL_LENGTH,
                  encoder_vocab_size=ACTION_SPACE + 1, # action_space + <SOS>
                  decoder_vocab_size=ACTION_SPACE + 1, # action_space + <SOS>
                  encoder_emb_size=5,
                  encoder_hidden_size=12,
                  decoder_hidden_size=12,
                  batch_size=UPDATE_BATCHSIZE,
                  num_cpu=CPU,
                  tensorboard_log=TENSORBOARD_LOGDIR,
                  input_processing=True
                  )
# ====================




    

# === Start NAO skill search ===
for iteration in range(NUM_ITERATION):


    MODEL_NAME = TENSORBOARD_LOGNAME + '_iter{}'.format(itereation)
    is_first_iteration = 0 == iteration

    # normalize scores
    norm_scores = min_max_normalization(X=scores,
                                        lower=0.2,
                                        upper=0.8
                                        )

    # fit (skills, norm_scores)
    model.learn(X=skills,
                y=norm_scores,
                epochs=EPOCHS,
                eval_interval=EVAL_INTERVAL,
                log_interval=LOG_INTERVAL,
                tb_log_name=MODEL_NAME,
                reset_num_timesteps=is_first_iteration)


    # get top N scored skills
    top_100_skills, top_100_scores = get_top_n(N=100,
                                               seqs=top_100_skills,
                                               scores=top_100_scores)

    # generate new skills
    new_skills, _ = model.predict(seeds=top_100_skills,
                                  lambdas=[10, 20, 30])

    div_skills = divide_skills(new_skills)

    new_scores = get_scores(div_skills)


    # === Print logs ===
    LOG.set_header('Iter {}/{}'.format(iteration+1, NUM_ITERATION)) 
    LOG.switch_group('Top 10 ranking (Old)')
    log_top_n_scores(10, skills, scores)
    LOG.switch_group('Top 10 ranking (New)')
    log_top_n_scores(10, new_skills, new_scores)

    skills.extend(new_skills)
    scores.extend(new_scores)

    LOG.switch_group('Top 10 ranking (Sum)')
    log_top_n_scores(10, skills, scores)

    LOG.dump_to_log(level=logging.INFO)
    # ==================

    # save skills
    output_skills(path=os.path.join(SKILL_SAVE_PATH, MODEL_NAME + '.skills'),
                  _skills=div_skills,
                  _scores=new_scores)

    # save model
    model.save(os.path.join(MODEL_SAVE_PATH, MODEL_NAME + '.model'))

# ==============================
