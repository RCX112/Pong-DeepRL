import numpy as np
import pickle
import gym
import tensorflow as tf
import time
import os
import gym_unrealcv
import matplotlib.pyplot as plt


gym_env = "jupong-3D-v0"
env_name = f"{gym_env}/pg_test"
reward_file_name = f"results/{env_name}/rewards1.txt"

if not os.path.exists(f"results"):
    os.mkdir(f"results")

if not os.path.exists(f"results/{gym_env}"):
    os.mkdir(f"results/{gym_env}")

if not os.path.exists(f"results/{env_name}"):
    os.mkdir(f"results/{env_name}")

log_path = f"results/{env_name}/log"
if not os.path.exists(log_path):
    os.mkdir(log_path)

checkpoint_path = f"results/{env_name}/log/checkpoints"
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
    
observation_path = f"results/{env_name}/observations"
if not os.path.exists(observation_path):
    os.mkdir(observation_path) 
 
def write_reward_into_file(file_name, episode_number, reward, env_name):
    if not os.path.exists(f"results/{env_name}"):
        os.mkdir(f"results/{env_name}")
    with open(file_name,'a') as f_rew:
                f_rew.write(f"episode: {episode_number}, reward: {reward}")
                f_rew.write("\n")

def get_reward_file_name():
    num = 1
    file_name_prefix = f"results/{env_name}/rewards"
    while True:
        file_name = f"{file_name_prefix}{num}.txt"
        if not os.path.exists(file_name):
            return file_name
        else:
            num += 1
            
def prepro(I):
    return env.prepro(I)

def discount_rewards(r):
    gamma = 0.99
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    # for t in reversed(range(0, r.size)):
    for t in reversed(range(0, len(r))):
        if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def load_model(path):
    model = pickle.load(open(path, 'rb'))
    return model['W1'].T, model['W2'].reshape((model['W2'].size,-1))

def make_network(pixels_num, hidden_units):
    pixels = tf.placeholder(dtype=tf.float32, shape=(None, pixels_num))    
    actions = tf.placeholder(dtype=tf.float32, shape=(None,1))
    rewards = tf.placeholder(dtype=tf.float32, shape=(None,1))

    with tf.variable_scope('policy'):
        hidden = tf.layers.dense(pixels, hidden_units, activation=tf.nn.relu,\
                kernel_initializer = tf.contrib.layers.xavier_initializer())
        hidden2 = tf.layers.dense(hidden, hidden_units, activation=tf.nn.relu,\
                kernel_initializer = tf.contrib.layers.xavier_initializer())
        logits = tf.layers.dense(hidden2, 1, activation=None,\
                kernel_initializer = tf.contrib.layers.xavier_initializer())

        out = tf.sigmoid(logits, name="sigmoid")
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=actions, logits=logits, name="cross_entropy")
        loss = tf.reduce_sum(tf.multiply(rewards, cross_entropy, name="rewards"))

    # lr=1e-4
    lr=1e-3
    decay_rate=0.99
    opt = tf.train.RMSPropOptimizer(lr, decay=decay_rate).minimize(loss)
    # opt = tf.train.AdamOptimizer(lr).minimize(loss)

    tf.summary.histogram("hidden_out", hidden)
    tf.summary.histogram("logits_out", logits)
    tf.summary.histogram("prob_out", out)
    merged = tf.summary.merge_all()

    # grads = tf.gradients(loss, [hidden_w, logit_w])
    # return pixels, actions, rewards, out, opt, merged, grads
    return pixels, actions, rewards, out, opt, merged

pixels_num = 110 * 110
hidden_units = 200
batch_size = 1

tf.reset_default_graph()
pix_ph, action_ph, reward_ph, out_sym, opt_sym, merged_sym = make_network(pixels_num, hidden_units)

resume = True
render = False
if not resume:
    reward_file_name = get_reward_file_name()

sess = tf.Session()
saver = tf.train.Saver()
writer = tf.summary.FileWriter(f'{log_path}/train', sess.graph)

if resume:
    try:
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    except ValueError:
        sess.run(tf.global_variables_initializer())
else:
    sess.run(tf.global_variables_initializer())

env = gym.make(gym_env)
observation = env.reset()

prev_x = None # used in computing the difference frame
xs = []
ys = []
ws = []
ep_ws = []
batch_ws = []
step = pickle.load(open(f'{log_path}/step.p', 'rb')) if resume and os.path.exists(f'{log_path}/step.p') else 0
episode_number = step*10
reward_mean = pickle.load(open(f'{log_path}/reward_mean.p', 'rb')) if resume and os.path.exists(f'{log_path}/reward_mean.p') else -20.0
while True:
    if render: env.render()
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros((pixels_num,))
    prev_x = cur_x
    
    #plt.imshow(x.reshape((110, 110)))
    #plt.show()

    assert x.size==pixels_num
    tf_probs = sess.run(out_sym, feed_dict={pix_ph:x.reshape((-1,x.size))})
    y = 1 if np.random.uniform() < tf_probs[0,0] else 0
    action = y
    del observation
    observation, reward, done, _ = env.step(action)

    xs.append(x)
    #ys.append(y)
    #ep_ws.append(reward)

    if done:
        del xs
        del ys
        del ep_ws
        
        xs = []
        ys = []
        ep_ws = []
        observation = env.reset()            

env.close()