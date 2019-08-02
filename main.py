import os
import sys
import time
import logging
import multiprocessing

# add library path
sys.path.append(os.path.abspath('./nao_search'))
sys.path.append(os.path.abspath('../lib'))

# nao search
from nao_search import epd
from nao_search.common.utils import random_sequences
from nao_search.common.utils import min_max_normalization
from nao_search.common.utils import pairwise_accuracy
from nao_search.common.utils import get_top_n

from nao_search.common.logger import Logger
from nao_search.common.logger import LoggingConfig

from nao_search.processes import UnsafePool

# openai gym
import gym

from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy

# atari
from env_wrapper import ActionRemapWrapper, SkillWrapper
from manager import AtariPolicyManager



# === use logging config ===
LoggingConfig.Use(filename='nao_skill_search_atari_(1, 9).train.log', 
                  output_to_file=True,
                  level='DEBUG')

LOG = Logger('nao_search')



# === Parameters ===
SKILL_LENGTH = 9
NUM_SKILLS_PER_SET = 1
ACTION_SPACE = 6

# === NAO ===
NUM_ITERATION = 5
UPDATE_BATCHSIZE = 128
CPU = 1
EPOCHS = 300
EVAL_INTERVAL = 1
LOG_INTERVAL = 1

# === Atari ===
TRAINING_STEPS = 5000
ENV_ID = 'Alien-ramDeterministic-v4'

# === Misc ===
MODEL_SAVE_PATH = './models'
SKILL_SAVE_PATH = './skills'
TENSORBOARD_LOGDIR = './epd_logs'
TENSORBOARD_LOGNAME = 'nao_skill_search_atari_({}, {})'.format(NUM_SKILLS_PER_SET, SKILL_LENGTH)
# ==================




# === Utility ===
def evaluate(skill):

    t_eval_start = time.time()
    env_creator = lambda: ActionRemapWrapper(gym.make(ENV_ID))
    atari_manager = AtariPolicyManager(env_creator=env_creator, 
                                       model=PPO2, 
                                       policy=MlpPolicy,
                                       save_path='alien',
                                       verbose=0,
                                       num_cpu=4)
    ave_score, ave_action_reward = atari_manager.get_rewards(skill, train_total_timesteps=TRAINING_STEPS)

    return ave_score, time.time()-t_eval_start

def get_scores(skills):
    global pool # process pool

    LOG.info('start evaluation')

    t_start = time.time()

    # evaluate scores of every skills using multiprocessing
    res = pool.map(evaluate, skills)
    
    _scores, _times = zip(*res)
    t_sum = sum(_times)
    t_mean = t_sum / len(_times)

    LOG.info('skill evaluation done, total: {:.6f} sec, mean: {:.6f} sec'.format(t_sum, t_mean))

    return list(_scores)
    


def make_dirs(path):
    import errno
    try:
        os.makedirs(path)
    except os.error as e:
        if e.errno != errno.EEXIST:
            raise


def divide_skills(skills):
    # divide long skills into skill sets

    def divide_skill(skill):
        
        assert len(skill) // NUM_SKILLS_PER_SET == SKILL_LENGTH

        avg = SKILL_LENGTH
        out = []
        last = 0

        while last < len(skill):
            out.append(skill[int(last):int(last + avg)])
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
    # print skills to file
    with open(path, 'w') as f:
        for skill, score in zip(_skills, _scores):
            f.write('{}:{}\n'.format(skill, score))
# ===============




# === Initialize ===
# prevent from recursive call by child processes
if __name__ != '__main__':
    exit()

# create dirs
make_dirs(MODEL_SAVE_PATH)
make_dirs(SKILL_SAVE_PATH)

# calculate the total length of flattened skill set
TOTAL_SKILL_LENGTH = SKILL_LENGTH * NUM_SKILLS_PER_SET
# set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
# create process pool
multiprocessing.set_start_method('fork')
pool = UnsafePool(16)
# ==================



# === Generate random skills ===
skills = random_sequences(
                        length=SKILL_LENGTH,
                        seq_num=30,
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
model = epd.BaseModel(
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


    MODEL_NAME = TENSORBOARD_LOGNAME + '_iter{}'.format(iteration)
    # check if it is the first iteration or not
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
    top_100_skills, top_100_scores = get_top_n(N=10,
                                               seqs=skills,
                                               scores=scores)

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
