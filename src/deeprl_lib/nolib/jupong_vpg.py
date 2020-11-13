"""
This module contains a simple example to train the Gym-Environments JuPong2D and JuPong3D with a Deep Reinforcement
Learning Algorithm. Here it is a Vanilla Policy Gradient Algorithm. The module was created to analyze which image
resolution is the most suitable for training a neuronal network. The quality is being measured by the CPU-time and the
return-values per episode.
"""
import gym, gym_pong
import numpy as np, os, time
import pickle, psutil, argparse
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import tensorflow as tf    


class JuPong_VPG:
    """
    The class 'JuPong_VPG' starts the training and analysis of the Gym-Environments JuPong2D and JuPong3D (Unreal
    Engine). The training of the model happens after 10 episodes.
    """
    def __init__(self, output, obs_size, folder_version, is_3D = False, game_system = "Linux", cam_angle = 0.0,
                 paddle_length = None, paddle_speed = None, ball_speed = None):
        """
        The constructor of this class creates all necessary configurations, which depend on the given parameters.
        :param output: The output folder for the neuronal network and the training results
        :param obs_size: Size of the observation, which is the input of the neuronal network
        :param folder_version: Session-ID for a specific training configuration
        :param is_3D: Boolean to play in the Gym-Environment JuPong3D (Unreal Engine)
        :param game_system: Operating System for the Gym-Environment JuPong3D
        :param cam_angle: Angle of the camera in JuPong3D. It rotates around the z-axis
        :param paddle_length: Factor for the paddle length
        :param paddle_speed: Factor for the paddle speed
        :param ball_speed: Factor for the ball speed
        """
        self.is_3D = is_3D
        self.ball_color = None
        self.output = output

        if not self.is_3D:    
            start_size = 400
            self.obs_size = obs_size
            self.zoom_val = obs_size / start_size
            img_size = start_size / self.STANDARD_SIZE_2D
            gym_env = f"jupong2d-headless-{img_size}-v3"
            env_name = f"{gym_env}/pg_{start_size}_{self.obs_size}_{folder_version}"
            
            if paddle_length is not None:
                env_name += f"_pl_{paddle_length}"
            if paddle_speed is not None:
                env_name += f"_ps_{paddle_speed}"
            if ball_speed is not None:
                env_name += f"_bs_{ball_speed}"
            
            if self.ball_color is not None:
                self.color_val = self.ball_color[0] * 0.5 + self.ball_color[1] * (2.0/6.0) + self.ball_color[2] * (1.0/6.0)
                self.step_size = int(start_size // self.obs_size)
        else:
            import gym_unrealcv
            self.obs_size = obs_size
            gym_env = f"jupong-3D-{game_system}-v0"
            env_name = f"{gym_env}/pg_{self.obs_size}_{cam_angle}_{folder_version}"
            
        
        if not os.path.exists(f"{output}"):
            os.mkdir(f"{output}")

        if not os.path.exists(f"{output}/{gym_env}"):
            os.mkdir(f"{output}/{gym_env}")

        if not os.path.exists(f"{output}/{env_name}"):
            os.mkdir(f"{output}/{env_name}")

        self.log_path = f"{output}/{env_name}/log"
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        self.checkpoint_path = f"{output}/{env_name}/log/checkpoints"
        if not os.path.exists(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        self.observation_path = f"{output}/{env_name}/observations"
        if not os.path.exists(self.observation_path):
            os.mkdir(self.observation_path)

        self.jupong_process = psutil.Process()

        self.cpu_time_secs_file = f"{output}/{env_name}/cpu_time_secs_1.csv"
        self.cpu_time_secs = self.load_array_from_file(self.cpu_time_secs_file, col_len = 2)

        self.reward_file_name = f"{output}/{env_name}/rewards1.csv"
        self.reward_arr = self.load_array_from_file(self.reward_file_name, col_len = 2)

        if not self.is_3D:
            self.ball_hits_file = f"{output}/{env_name}/ball_hits1.csv"
            self.ball_hits_arr = self.load_array_from_file(self.ball_hits_file, col_len = 3)

            self.ball_misses_file = f"{output}/{env_name}/ball_misses1.csv"
            self.ball_misses_arr = self.load_array_from_file(self.ball_misses_file, col_len = 3)

            self.ball_crossing_file = f"{output}/{env_name}/ball_crossing1.csv"
            self.ball_crossing_arr = self.load_array_from_file(self.ball_crossing_file, col_len = 3)
            
            self.agent_angles_file = f"{output}/{env_name}/ball_angles_agent.csv"
            self.agent_angles_arr = self.load_array_from_file(self.agent_angles_file, col_len = 2)
            
            self.cpu_angles_file = f"{output}/{env_name}/ball_angles_cpu.csv"
            self.cpu_angles_arr = self.load_array_from_file(self.cpu_angles_file, col_len = 2)
            
            self.ball_dir_vel_file = f"{output}/{env_name}/ball_dir_and_vel_agent.csv"
            self.ball_dir_vel_arr = self.load_array_from_file(self.ball_dir_vel_file , col_len = 4)
        
        self.pixels_num = self.obs_size * self.obs_size
        self.hidden_units = 200
        self.batch_size = 10
        
        tf.reset_default_graph()
        self.pix_ph, self.action_ph, self.reward_ph, self.out_sym, self.opt_sym, self.merged_sym = self.make_network(self.pixels_num, self.hidden_units)

        resume = True

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(f'{self.log_path}/train', self.sess.graph)

        if resume:
            try:
                self.saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_path))
            except ValueError:
                self.sess.run(tf.global_variables_initializer())
        else:
            self.sess.run(tf.global_variables_initializer())

        self.env = gym.make(gym_env)
        if self.ball_color is not None:
            self.env.set_ball_color(self.ball_color)
        if self.is_3D:
            self.env.change_camera_angle(cam_angle)
        if paddle_length is not None:
            self.env.scale_paddle_height(paddle_length)
        if paddle_speed is not None:
            self.env.scale_paddle_vel(paddle_speed)
        if ball_speed is not None:
            self.env.scale_ball_velocity(ball_speed)
        
        self.observation = self.env.reset()

        self.step = pickle.load(open(f'{self.log_path}/step.p', 'rb')) if resume and os.path.exists(f'{self.log_path}/step.p') else 0   
        self.reward_mean = pickle.load(open(f'{self.log_path}/reward_mean.p', 'rb')) if resume and os.path.exists(f'{self.log_path}/reward_mean.p') else -20.0


    @property
    def STANDARD_SIZE_2D(self):
        """
        Standard size of the JuPong2D-Image
        """
        return 400

    def write_ball_angles_into_file(self, file_name, angles):
        """
        Writes the angles of the ball colliding with a paddle into a file.
        :param file_name: Path to the file
        :param angles: Ball angles
        """
        with open(file_name, 'a') as f_angles:
            for angle in angles:
                f_angles.write(f"{angle}")
                f_angles.write("\n")

    def prepro(self, I):
        """
        This method creates the observations by preprocessing the image from the Gym-Environment.
        :param I: Current image from the Gym-Environment
        """
        if not self.is_3D:
            if self.ball_color is not None:
                I = I[::self.step_size, ::self.step_size, :]
                ball_index_arr = []
                for i in range(I.shape[0]):
                    for j in range(I.shape[1]):
                        if (I[i, j] == self.ball_color).all():
                            ball_index_arr.append([i, j])
                I = I[:, :, 0]
                return I.astype(np.float), ball_index_arr
            else:
                I = zoom(I[:, :, 0], self.zoom_val)
                I[I < 128] = 0
                I[I != 0] = 1

                return I.astype(np.float).ravel()
        else:
            return self.env.prepro(I)

    def discount_rewards(self, r):
        """
        Take 1D float array of rewards and compute discounted reward.
        :param r: Array of rewards
        """
        gamma = 0.99
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, len(r))):
            if r[t] != 0: running_add = 0
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r

    def load_model(self, path):
        """
        Loads a saved model from a given path.
        :param path: Path to the model
        """
        model = pickle.load(open(path, 'rb'))
        return model['W1'].T, model['W2'].reshape((model['W2'].size,-1))

    def make_network(self, pixels_num, hidden_units):
        """
        Creates a neuronal network with TensorFlow. It has an input layer, two hidden layers and an output layer. It
        calculates a probability distribution for the next action of a current observation.
        :param pixels_num: Size of the input layer
        :param hidden_units: Sizes of the hidden layers
        """
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

        lr=1e-3
        decay_rate=0.99
        opt = tf.train.RMSPropOptimizer(lr, decay=decay_rate).minimize(loss)

        tf.summary.histogram("hidden_out", hidden)
        tf.summary.histogram("logits_out", logits)
        tf.summary.histogram("prob_out", out)
        merged = tf.summary.merge_all()

        return pixels, actions, rewards, out, opt, merged

    def get_cpu_time(self, process):
        """
        Method to calculate the CPU-time of the running training-process.
        :param process: Training-process
        """
        return sum(process.cpu_times()[:2])

    def save_array_to_csv(self, arr, file):
        """
        Saves an array into a csv-file.
        :param arr: Array to save
        :param file: Path of the csv-file
        """
        np.savetxt(file, np.asarray(arr), delimiter=",")

    def load_array_from_csv(self, file, to_list = False):
        """
        Loads an array from a csv-file.
        :param file: Path to the csv-file
        :param to_list: Boolean to cast the array into a list. If it's False, the array will be a numpy array.
        """
        if to_list:
            return np.ndarray.tolist(np.genfromtxt(file, delimiter=','))
        else:
            return np.genfromtxt(file, delimiter=',')

    def load_array_from_file(self, file_name, to_list = False, col_len = None):
        """
        Loads an array from a file.
        :param file_name: Path to the file
        :param to_list: Boolean to cast the array into a list. If it's False, the array will be a numpy array.
        :param col_len: Integer to create an empty array with a specific column size
        """
        if not os.path.exists(file_name):
            if to_list:
                return []
            else:
                if col_len is None:
                    return np.array([])
                else:
                    return np.array([]).reshape((0, col_len))
        else:
            return self.load_array_from_csv(file_name, to_list)

    def start_training(self):
        """
        Starts the training and analysis of a neuronal network in one of the possible Gym-Environments
        """
        prev_x = None
        prev_ball_index_arr = None
        xs = []
        ys = []
        ws = []
        ep_ws = []
        batch_ws = []
        episode_number = self.step*10
        cpu_time1 = self.get_cpu_time(self.jupong_process)
        cpu_time2 = 0.0
        
        while True:
            if self.ball_color is not None:
                cur_x, ball_index_arr = self.prepro(self.observation)
                x = cur_x - prev_x if prev_x is not None else np.zeros((self.obs_size, self.obs_size))

                for ball_index in ball_index_arr:
                    x[ball_index[0], ball_index[1]] = self.color_val

                if prev_ball_index_arr is not None:
                    for ball_index in prev_ball_index_arr:
                        x[ball_index[0], ball_index[1]] = -self.color_val

                prev_x = cur_x
                prev_ball_index_arr = ball_index_arr
                x = x.ravel()
            else:
                cur_x = self.prepro(self.observation)
                x = cur_x - prev_x if prev_x is not None else np.zeros((self.pixels_num,))
                prev_x = cur_x

            #plt.imshow(x.reshape((self.obs_size, self.obs_size)))
            #print(x[x!=0])
            #plt.show()
            
            assert x.size == self.pixels_num
            tf_probs = self.sess.run(self.out_sym, feed_dict={self.pix_ph:x.reshape((-1,x.size))})
            y = 1 if np.random.uniform() < tf_probs[0,0] else 0
            action = y
            del self.observation
            self.observation, reward, done, _ = self.env.step(action)
            
            xs.append(x)
            ys.append(y)
            ep_ws.append(reward)

            if done:
                episode_number += 1

                cpu_time2 = self.get_cpu_time(self.jupong_process)
                #print(f"CPU time: {cpu_time2 - cpu_time1}")

                self.cpu_time_secs = np.r_[self.cpu_time_secs, [[episode_number, (cpu_time2 - cpu_time1)]]]
                self.reward_arr = np.r_[self.reward_arr, [[episode_number, self.reward_mean]]]
                
                if not self.is_3D:
                    self.ball_hits_arr = np.r_[self.ball_hits_arr, [[episode_number, self.env.get_ball_hits()[0]
                                                                        , self.env.get_ball_hits()[1]]]]
                    self.ball_misses_arr = np.r_[self.ball_misses_arr, [[episode_number, self.env.get_ball_misses()[0]
                                                                            , self.env.get_ball_misses()[1]]]]
                    self.ball_crossing_arr = np.r_[self.ball_crossing_arr, [[episode_number,
                                                                             self.env.get_ball_crossing_sides()[0],
                                                                             self.env.get_ball_crossing_sides()[1]]]]
                    for agent_ball_angle in self.env.get_agent_ball_angles():
                        self.agent_angles_arr = np.r_[self.agent_angles_arr, [[episode_number, agent_ball_angle]]]
                    for cpu_ball_angle in self.env.get_cpu_ball_angles():
                        self.cpu_angles_arr = np.r_[self.cpu_angles_arr, [[episode_number, cpu_ball_angle]]]
                    for ball_dir_vel in self.env.get_agent_ball_dir_and_vel():
                        self.ball_dir_vel_arr = np.r_[self.ball_dir_vel_arr, [np.append(episode_number, ball_dir_vel)]]

                discounted_epr = self.discount_rewards(ep_ws)
                discounted_epr -= np.mean(discounted_epr)
                discounted_epr /= np.std(discounted_epr)
                batch_ws += discounted_epr.tolist()
                print(f"{episode_number}, reward: {sum(ep_ws)}")
                self.reward_mean = 0.99*self.reward_mean+(1-0.99)*(sum(ep_ws))
                del ep_ws
                ep_ws = []

                if self.reward_mean >= 20.0:
                    break

                if episode_number % self.batch_size == 0:
                    self.save_array_to_csv(self.cpu_time_secs, self.cpu_time_secs_file)
                    self.save_array_to_csv(self.reward_arr, self.reward_file_name)
                    if not self.is_3D:
                        self.save_array_to_csv(self.ball_hits_arr, self.ball_hits_file)
                        self.save_array_to_csv(self.ball_misses_arr, self.ball_misses_file)
                        self.save_array_to_csv(self.ball_crossing_arr, self.ball_crossing_file)
                        self.save_array_to_csv(self.agent_angles_arr, self.agent_angles_file)
                        self.save_array_to_csv(self.cpu_angles_arr, self.cpu_angles_file)
                        self.save_array_to_csv(self.ball_dir_vel_arr, self.ball_dir_vel_file)
                    
                    self.step += 1
                    exs = np.vstack(xs)
                    eys = np.vstack(ys)
                    ews = np.vstack(batch_ws)
                    frame_size = len(xs)
                    del xs
                    del ys
                    del discounted_epr
                    del batch_ws

                    stride = 20000
                    pos = 0
                    while True:
                        end = frame_size if pos+stride>=frame_size else pos+stride
                        batch_x = exs[pos:end]
                        batch_y = eys[pos:end]
                        batch_w = ews[pos:end]
                        tf_opt, tf_summary = self.sess.run([self.opt_sym, self.merged_sym],
                                                           feed_dict={self.pix_ph:batch_x, self.action_ph:batch_y,
                                                                      self.reward_ph:batch_w})
                        pos = end
                        if pos >= frame_size:
                            break
                    xs = []
                    ys = []
                    batch_ws = []
                    del exs
                    del eys
                    del ews
                    del batch_x
                    del batch_y
                    del batch_w
                    self.saver.save(self.sess, f"{self.checkpoint_path}/pg_{self.step}.ckpt")
                    self.writer.add_summary(tf_summary, self.step)
                    print("datetime: {}, episode: {}, update step: {}, frame size: {}, reward: {}".\
                            format(time.strftime('%X %x %Z'), episode_number, self.step, frame_size, self.reward_mean))
                    fp = open(f'{self.log_path}/step.p', 'wb')
                    pickle.dump(self.step, fp)
                    fp.close()
                    fp = open(f'{self.log_path}/reward_mean.p', 'wb')
                    pickle.dump(self.reward_mean, fp)
                    fp.close()

                self.observation = self.env.reset()       
                cpu_time1 = self.get_cpu_time(self.jupong_process)

        self.env.close()
        
        
def start():
    """
    Start-method of the VPG-training and analysis of the image resolution.
    usage: python jupong_vpg.py [-h] [--output OUTPUT] [--obs_size OBS_SIZE]
                                [--folder_version FOLDER_VERSION] [--is_3D]
                                [--system SYSTEM] [--angle ANGLE] [--pl PL] [--ps PS]
                                [--bs BS]

    optional arguments:
        -h, --help            show this help message and exit
        --output OUTPUT       Path to the results folder of the given environment.
        --obs_size OBS_SIZE   Length and width of the input image.
        --folder_version FOLDER_VERSION
                            Version of the folder.
        --is_3D               Boolean if playing in JuPong3D.
        --system SYSTEM       System of the Unreal Engine Environment.
        --angle ANGLE         Camera angle of the Unreal Engine Environment.
        --pl PL               Factor of the paddle-length.
        --ps PS               Factor of the paddle-speed.
        --bs BS               Factor of the ball-speed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="results",
                        help='Path to the results folder of the given environment.')
    parser.add_argument("--obs_size", type=int, default=60, help="Length and width of the input image.")
    parser.add_argument("--folder_version", type=str, default="testing", help="Version of the folder.")
    parser.add_argument("--is_3D", action='store_true', help="Boolean if playing in JuPong3D.")
    parser.add_argument("--system", type=str, default="Linux", help="System of the Unreal Engine Environment.")
    parser.add_argument("--angle", type=float, default=0.0, help="Camera angle of the Unreal Engine Environment.")
    parser.add_argument("--pl", type=float, default=None, help="Factor of the paddle-length.")
    parser.add_argument("--ps", type=float, default=None, help="Factor of the paddle-speed.")
    parser.add_argument("--bs", type=float, default=None, help="Factor of the ball-speed.")
    args = parser.parse_args()    
        
    jupong_agent = JuPong_VPG(args.output, args.obs_size, args.folder_version, is_3D = args.is_3D,
                              game_system = args.system, cam_angle = args.angle, paddle_length = args.pl,
                              paddle_speed = args.ps, ball_speed = args.bs)
    jupong_agent.start_training()
  
if __name__ == "__main__":
    start()
