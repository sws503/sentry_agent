from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from tqdm import tqdm

import sys
np.set_printoptions(threshold=sys.maxsize)

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index


# TODO: preprocessing functions for the following layers
_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

#action_list = [actions.FUNCTIONS.no_op.id, actions.FUNCTIONS.Attack_screen.id, actions.FUNCTIONS.Move_screen.id, actions.FUNCTIONS.HoldPosition_quick.id]
#[0, 12, 331, 274, 193]

#_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
#_SELECT_POINT = actions.FUNCTIONS.select_point.id

action_list = [193]

def preprocess_minimap(minimap):
  layers = []
  assert minimap.shape[0] == len(features.MINIMAP_FEATURES)
  for i in range(len(features.MINIMAP_FEATURES)):
    if i == _MINIMAP_PLAYER_ID:
      layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
    elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(minimap[i:i+1] / features.MINIMAP_FEATURES[i].scale)
    else:
      layer = np.zeros([features.MINIMAP_FEATURES[i].scale, minimap.shape[1], minimap.shape[2]], dtype=np.float32)
      for j in range(features.MINIMAP_FEATURES[i].scale):
        indy, indx = (minimap[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)

def preprocess_screen(screen):
  layers = []
  assert screen.shape[0] == len(features.SCREEN_FEATURES)
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      layers.append(screen[i:i+1] / features.SCREEN_FEATURES[i].scale)
    else:
      layer = np.zeros([features.SCREEN_FEATURES[i].scale, screen.shape[1], screen.shape[2]], dtype=np.float32)
      for j in range(features.SCREEN_FEATURES[i].scale):
        indy, indx = (screen[i] == j).nonzero()
        layer[j, indy, indx] = 1
      layers.append(layer)
  return np.concatenate(layers, axis=0)

def minimap_channel():
  c = 0
  for i in range(len(features.MINIMAP_FEATURES)):
    if i == _MINIMAP_PLAYER_ID:
      c += 1
    elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
      c += 1
    else:
      c += features.MINIMAP_FEATURES[i].scale
  return c

def screen_channel():
  c = 0
  for i in range(len(features.SCREEN_FEATURES)):
    if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
      c += 1
    elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
      c += 1
    else:
      c += features.SCREEN_FEATURES[i].scale
  return c


initial_eps = [0.8, 1.0]

class ZergAgent(base_agent.BaseAgent):
    def __init__(self, training):
        super(ZergAgent, self).__init__()

        #self.name = name
        self.training = training
        self.summary = []
        # Minimap size, screen size and info size
        #assert msize == ssize
        self.msize = 64
        #self.ssize = 64
        self.ssize = 84
        self.isize = len(actions.FUNCTIONS)

        self.epsilon_a = 0.5
        self.epsilon_b = 1.0
        #self.epsilon = [0.8, 1.0] #initial_eps
        self.spatial_list = []
        self.x = 0
        self.y = 0
        self.first = 0
        self.direction = []

        #self.msize = 64 # 미니맵 사이즈 - agent에서 관장
        #self.ssize = 84 # 스크린 사이즈 - agent에서 관장
        #self.isize = len(actions.FUNCTIONS)
        #TODO : infomation을 써야할까요?

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def sess(self, sess, summary_writer):
        self.sess = sess
        self.summary_writer = summary_writer

    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def reset(self):
        # Epsilon schedule
        #if self.epsilon_a > 0.001:
            #self.epsilon_a -= 0.001

        self.first = 0

        if self.epsilon_b > 0.01 :
            self.epsilon_b -= 0.0005
        print("epsilon b : ", self.epsilon_b)
        #print("epsilon a : ", self.epsilon_a, "\n b : ", self.epsilon_b)


        #self.epsilon = [a, b]

    def build_net(self, minimap, screen, info, msize, ssize, num_action):
        mconv1 = layers.conv2d(tf.transpose(minimap, [0, 2, 3, 1]),
                               num_outputs=16,
                               kernel_size=8,
                               stride=4,
                               scope='mconv1')
        mconv2 = layers.conv2d(mconv1,
                               num_outputs=32,
                               kernel_size=4,
                               stride=2,
                               scope='mconv2')
        sconv1 = layers.conv2d(tf.transpose(screen, [0, 2, 3, 1]),
                               num_outputs=16,
                               kernel_size=8,
                               stride=4,
                               scope='sconv1')
        sconv2 = layers.conv2d(sconv1,
                               num_outputs=32,
                               kernel_size=4,
                               stride=2,
                               scope='sconv2')
        info_fc = layers.fully_connected(layers.flatten(info),
                                   num_outputs=256,
                                   activation_fn=tf.tanh,
                                   scope='info_fc')


        # Compute spatial actions, non spatial actions and value
        feat_fc = tf.concat([layers.flatten(mconv2), layers.flatten(sconv2), info_fc], axis=1)
        feat_fc = layers.fully_connected(feat_fc,
                                         num_outputs=256,
                                         activation_fn=tf.nn.relu,
                                         scope='feat_fc')
        spatial_action_x = layers.fully_connected(feat_fc,
                                                  num_outputs=ssize,
                                                  activation_fn=tf.nn.softmax,
                                                  scope='spatial_action_x')
        spatial_action_y = layers.fully_connected(feat_fc,
                                                  num_outputs=ssize,
                                                  activation_fn=tf.nn.softmax,
                                                  scope='spatial_action_y')

        spatial_action_x = tf.reshape(spatial_action_x, [-1, 1, ssize])
        spatial_action_x = tf.tile(spatial_action_x, [1, ssize, 1])
        spatial_action_y = tf.reshape(spatial_action_y, [-1, ssize, 1])
        spatial_action_y = tf.tile(spatial_action_y, [1, 1, ssize])
        spatial_action = layers.flatten(spatial_action_x * spatial_action_y)

        non_spatial_action = layers.fully_connected(feat_fc,
                                                    num_outputs=num_action,
                                                    activation_fn=tf.nn.softmax,
                                                    scope='non_spatial_action')
        value = tf.reshape(layers.fully_connected(feat_fc,
                                                  num_outputs=1,
                                                  activation_fn=None,
                                                  scope='value'), [-1])

        return spatial_action, non_spatial_action, value

    def build_model(self, reuse, dev):
        with tf.variable_scope('Sentry_agent') and tf.device(dev):
            if reuse:
                tf.get_variable_scope().reuse_variables()
                assert tf.get_variable_scope().reuse

            # Set inputs of networks
            self.minimap = tf.placeholder(tf.float32, [None, minimap_channel(), self.msize, self.msize],
                                          name='minimap')
            self.screen = tf.placeholder(tf.float32, [None, screen_channel(), self.ssize, self.ssize], name='screen')
            self.info = tf.placeholder(tf.float32, [None, self.isize], name='info')

            # Build networks
            net = self.build_net(self.minimap, self.screen, self.info, self.msize, self.ssize, len(actions.FUNCTIONS))
            self.spatial_action, self.non_spatial_action, self.value = net

            # Set targets and masks
            self.valid_spatial_action = tf.placeholder(tf.float32, [None], name='valid_spatial_action')
            self.spatial_action_selected = tf.placeholder(tf.float32, [None, self.ssize ** 2],
                                                          name='spatial_action_selected')
            self.valid_non_spatial_action = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                                           name='valid_non_spatial_action')
            self.non_spatial_action_selected = tf.placeholder(tf.float32, [None, len(actions.FUNCTIONS)],
                                                              name='non_spatial_action_selected')
            self.value_target = tf.placeholder(tf.float32, [None], name='value_target')

            # Compute log probability
            spatial_action_prob = tf.reduce_sum(self.spatial_action * self.spatial_action_selected, axis=1)
            spatial_action_log_prob = tf.log(tf.clip_by_value(spatial_action_prob, 1e-10, 1.))
            non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.non_spatial_action_selected, axis=1)
            valid_non_spatial_action_prob = tf.reduce_sum(self.non_spatial_action * self.valid_non_spatial_action,
                                                          axis=1)
            valid_non_spatial_action_prob = tf.clip_by_value(valid_non_spatial_action_prob, 1e-10, 1.)
            non_spatial_action_prob = non_spatial_action_prob / valid_non_spatial_action_prob
            non_spatial_action_log_prob = tf.log(tf.clip_by_value(non_spatial_action_prob, 1e-10, 1.))
            self.summary.append(tf.summary.histogram('spatial_action_prob', spatial_action_prob))
            self.summary.append(tf.summary.histogram('non_spatial_action_prob', non_spatial_action_prob))

            # Compute losses, more details in https://arxiv.org/abs/1602.01783

            # Policy loss and value loss
            action_log_prob = self.valid_spatial_action * spatial_action_log_prob + non_spatial_action_log_prob
            advantage = tf.stop_gradient(self.value_target - self.value)
            policy_loss = - tf.reduce_mean(action_log_prob * advantage)
            value_loss = - tf.reduce_mean(self.value * advantage)
            self.summary.append(tf.summary.scalar('policy_loss', policy_loss))
            self.summary.append(tf.summary.scalar('value_loss', value_loss))

            # TODO: policy penalty
            loss = policy_loss + value_loss

            # Build the optimizer
            self.learning_rate = tf.placeholder(tf.float32, None, name='learning_rate')
            opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.99, epsilon=1e-10)
            grads = opt.compute_gradients(loss)
            cliped_grad = []
            for grad, var in grads: #gradient, variable pair
                self.summary.append(tf.summary.histogram(var.op.name, var))
                self.summary.append(tf.summary.histogram(var.op.name + '/grad', grad))
                grad = tf.clip_by_norm(grad, 10.0)
                cliped_grad.append([grad, var])
            self.train_op = opt.apply_gradients(cliped_grad)
            self.summary_op = tf.summary.merge(self.summary)

            self.saver = tf.train.Saver(max_to_keep=100)

        # TODO: policy penalty
        loss = policy_loss + value_loss

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]


    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def get_spatial_list(self):
        return self.spatial_list

    def step_run(self, obs):
        minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
        minimap = np.expand_dims(preprocess_minimap(minimap), axis=0)
        screen = np.array(obs.observation.feature_screen, dtype=np.float32)
        screen = np.expand_dims(preprocess_screen(screen), axis=0)
        # TODO: only use available actions
        info = np.zeros([1, self.isize], dtype=np.float32)
        info[0, obs.observation['available_actions']] = 1

        feed = {self.minimap: minimap,
                self.screen: screen,
                self.info: info}
        non_spatial_action, spatial_action = self.sess.run(
            [self.non_spatial_action, self.spatial_action],
            feed_dict=feed)

        # Select an action and a spatial target
        non_spatial_action = non_spatial_action.ravel()

        spatial_action = spatial_action.ravel()
        valid_actions = np.array(action_list)

        self.spatial_list = spatial_action

        #print(spatial_action)
        #print(" ")

        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        target = np.argmax(spatial_action)
        target = [int(target // self.ssize), int(target % self.ssize)]

        # Epsilon greedy exploration
        if self.training and np.random.rand() < self.epsilon_a:
            act_id = np.random.choice(valid_actions)
        if self.training and np.random.rand() < self.epsilon_b:
            dy = np.random.randint(10, 50)
            target[0] = int(max(0, min(self.ssize - 1, dy)))
            dx = np.random.randint(20, 60)
            target[1] = int(max(0, min(self.ssize - 1, dx)))
        else :
            print('greedy target x: {}, y: {}'.format(target[1], target[0]))

        # Set act_id and act_args
        act_args = []
        #act_id = 193 #TODO : 임시방편으로 역장으로 act_id를 고정
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([target[1], target[0]])
            else:
                act_args.append([0])  # TODO: Be careful
        return act_id, act_args

    def step(self, obs):
        super(ZergAgent, self).step(obs)

        armies = [unit for unit in obs.observation.feature_units
                  if unit.alliance == features.PlayerRelative.SELF]
        enemies = [unit for unit in obs.observation.feature_units
                   if unit.alliance == features.PlayerRelative.ENEMY]

        if self.first == 0:
            act_id, act_args = self.step_run(obs)
            self.first = 1
            self.x = act_args[1][0]
            self.y = act_args[1][1]
            self.direction = act_args.copy()
            print("direction : ", self.direction)
            psi_grid = [[0], [self.direction[1][0] + 2, self.direction[1][1]]]
            print("psi direction : ", psi_grid)

        elif self.first == 1:
            grid = (self.x, self.y)

            if self.can_do(obs, actions.FUNCTIONS.Attack_screen.id):
                for army in armies:
                    if self.can_do(obs, actions.FUNCTIONS.Effect_PsiStorm_screen.id) and army.energy >= 75 :
                        #act_id, act_args = self.step_run(obs)
                        psi_grid = [[0], [self.direction[1][0] + 3, self.direction[1][1]]]
                        return actions.FunctionCall(218, psi_grid)

                    elif self.can_do(obs, actions.FUNCTIONS.Effect_ForceField_screen.id) and army.energy >= 50 :
                        #act_id, act_args = self.step_run(obs)

                        return actions.FunctionCall(193, self.direction)

                    elif self.can_do(obs, actions.FUNCTIONS.Attack_screen.id) and len(enemies) > 0:
                        enemy = random.choice(enemies)
                        # return actions.FUNCTIONS.Attack_minimap("queued", (0, 32))
                        return actions.FUNCTIONS.Attack_minimap("queued", (55, 25))#(enemy.x, enemy.y))

                    else:
                        print("error occur")
                        return actions.FUNCTIONS.no_op()

        elif self.can_do(obs, actions.FUNCTIONS.select_army.id) :
             return actions.FUNCTIONS.select_army("select")

        return actions.FUNCTIONS.no_op()

    def update(self, rbs, disc, lr, cter):
        print("학습을 시작합니다!")
        # Compute R, which is value of the last observation
        obs = rbs[-1][-1]
        if obs.last():
            R = 0
        else:
            minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
            minimap = np.expand_dims(preprocess_minimap(minimap), axis=0)
            screen = np.array(obs.observation.feature_screen, dtype=np.float32)
            screen = np.expand_dims(preprocess_screen(screen), axis=0)
            info = np.zeros([1, self.isize], dtype=np.float32)
            info[0, obs.observation['available_actions']] = 1

            feed = {self.minimap: minimap,
                    self.screen: screen,
                    self.info: info}
            R = self.sess.run(self.value, feed_dict=feed)[0]

        # Compute targets and masks
        minimaps = []
        screens = []
        infos = []

        value_target = np.zeros([len(rbs)], dtype=np.float32)
        value_target[-1] = R

        valid_spatial_action = np.zeros([len(rbs)], dtype=np.float32)
        spatial_action_selected = np.zeros([len(rbs), self.ssize ** 2], dtype=np.float32)
        valid_non_spatial_action = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)
        non_spatial_action_selected = np.zeros([len(rbs), len(actions.FUNCTIONS)], dtype=np.float32)

        rbs.reverse()
        for i, [obs, action, next_obs] in enumerate(tqdm(rbs)):
            minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
            minimap = np.expand_dims(preprocess_minimap(minimap), axis=0)
            screen = np.array(obs.observation.feature_screen, dtype=np.float32)
            screen = np.expand_dims(preprocess_screen(screen), axis=0)
            info = np.zeros([1, self.isize], dtype=np.float32)
            info[0, obs.observation['available_actions']] = 1

            minimaps.append(minimap)
            screens.append(screen)
            infos.append(info)

            score = obs.observation["score_cumulative"][0]
            #print("score는 ", score)
            #reward = obs.reward
            #print("reward는 ", reward)
            act_id = action.function
            act_args = action.arguments

            value_target[i] = score + disc * value_target[i - 1]
            #value_target[i] = reward + disc * value_target[i - 1]

            valid_actions = np.array(action_list)#obs.observation["available_actions"]
            valid_non_spatial_action[i, valid_actions] = 1
            non_spatial_action_selected[i, act_id] = 1

            args = actions.FUNCTIONS[act_id].args
            for arg, act_arg in zip(args, act_args):
                if arg.name in ('screen', 'minimap', 'screen2'):
                    ind = act_arg[1] * self.ssize + act_arg[0]
                    valid_spatial_action[i] = 1
                    spatial_action_selected[i, ind] = 1

        minimaps = np.concatenate(minimaps, axis=0)
        screens = np.concatenate(screens, axis=0)
        infos = np.concatenate(infos, axis=0)

        # Train
        feed = {self.minimap: minimaps,
                self.screen: screens,
                self.info: infos,
                self.value_target: value_target,
                self.valid_spatial_action: valid_spatial_action,
                self.spatial_action_selected: spatial_action_selected,
                self.valid_non_spatial_action: valid_non_spatial_action,
                self.non_spatial_action_selected: non_spatial_action_selected,
                self.learning_rate: lr}
        _, summary = self.sess.run([self.train_op, self.summary_op], feed_dict=feed)
        self.summary_writer.add_summary(summary, cter)

    def save_model(self, path, count):
        self.saver.save(self.sess, path+'/model.pkl', count)


    def load_model(self, path):
        print("모델을 불러옵니다!")
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        return int(ckpt.model_checkpoint_path.split('-')[-1])
