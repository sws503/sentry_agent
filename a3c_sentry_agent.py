from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index


# TODO: preprocessing functions for the following layers
_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index
_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index


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




class ZergAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ZergAgent, self).__init__()

        #self.name = name
        self.training = True
        self.summary = []
        # Minimap size, screen size and info size
        #assert msize == ssize
        self.msize = 64
        #self.ssize = 64
        self.ssize = 84
        self.isize = len(actions.FUNCTIONS)

        #self.msize = 64 # 미니맵 사이즈 - agent에서 관장
        #self.ssize = 84 # 스크린 사이즈 - agent에서 관장
        #self.isize = len(actions.FUNCTIONS)
        #TODO : infomation을 써야할까요?

    def setup(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def sess(self, sess):
        self.sess = sess
        #self.summary_writer = summary_writer

    def initialize(self):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

    def reset(self):
        # Epsilon schedule
        self.epsilon = [0.05, 0.2]

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

    def build_model(self, reuse):
        with tf.device('/cpu:0'):
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
            for grad, var in grads:
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

    def step_run(self, obs):
        minimap = np.array(obs.observation.feature_minimap, dtype=np.float32)
        minimap = np.expand_dims(preprocess_minimap(minimap), axis=0)
        screen = np.array(obs.observation.feature_screen, dtype=np.float32)
        screen = np.expand_dims(preprocess_screen(screen), axis=0)
        # TODO: only use available actions
        info = np.zeros([1, self.isize], dtype=np.float32)
        info[0, obs.observation['available_actions']] = 1

        #print("minimap",minimap)
        #print("screen",screen)

        feed = {self.minimap: minimap,
                self.screen: screen,
                self.info: info}
        non_spatial_action, spatial_action = self.sess.run(
            [self.non_spatial_action, self.spatial_action],
            feed_dict=feed)

        # Select an action and a spatial target
        non_spatial_action = non_spatial_action.ravel()

        spatial_action = spatial_action.ravel()
        print("spatial_action : ", spatial_action)
        valid_actions = obs.observation['available_actions']

        act_id = valid_actions[np.argmax(non_spatial_action[valid_actions])]
        target = np.argmax(spatial_action)
        print("target : ", target)
        target = [int(target // self.ssize), int(target % self.ssize)]
        print("target : ", target)

        # Epsilon greedy exploration
        if self.training and np.random.rand() < self.epsilon[0]:
            act_id = np.random.choice(valid_actions)
        if self.training and np.random.rand() < self.epsilon[1]:
            dy = np.random.randint(-4, 5)
            target[0] = int(max(0, min(self.ssize - 1, target[0] + dy)))
            dx = np.random.randint(-4, 5)
            target[1] = int(max(0, min(self.ssize - 1, target[1] + dx)))

        # Set act_id and act_args
        act_args = []
        act_id = 193 #TODO : 임시방편으로 역장으로 act_id를 고정
        for arg in actions.FUNCTIONS[act_id].args:
            if arg.name in ('screen', 'minimap', 'screen2'):
                act_args.append([target[1], target[0]])
            else:
                act_args.append([0])  # TODO: Be careful
        return act_id, act_args



    def step(self, obs):
        super(ZergAgent, self).step(obs)


        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            xmean = player_x.mean()
            ymean = player_y.mean()


        hydralisks = self.get_units_by_type(obs, units.Zerg.Hydralisk)
        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
        Sentries = self.get_units_by_type(obs, units.Protoss.Sentry)

        if self.can_do(obs, actions.FUNCTIONS.Effect_ForceField_screen.id):
            act_id, act_args = self.step_run(obs)
            #TODO : 지금은 무조건 역장으로
            #screen = self.ssize
            #target = [random.randint(0, screen),random.randint(0, screen)]
            #TODO : target을 학습시키는 agent를 만드는 것이 목표
            #return actions.FUNCTIONS.Effect_ForceField_screen("now", target)

            print("제가 고른 act_id는 ", act_id)
            print("제가 고른 act_args는 ", act_args)
            return actions.FunctionCall(act_id, act_args)
            #return actions.FUNCTIONS.no_op()

        elif self.can_do(obs, actions.FUNCTIONS.Attack_screen.id) :

            if len(hydralisks)>0:
                enemy = random.choice(hydralisks)
            elif len(zerglings)>0:
                enemy = random.choice(zerglings)
            else :
                return actions.FUNCTIONS.no_op()
            #return actions.FUNCTIONS.Attack_minimap("queued", (0, 32))
            return actions.FUNCTIONS.Attack_minimap("queued",(enemy.x, enemy.y))
                #ISSUE : Attack_minimap이 불가능한 상황에서 명령을 실행함

        elif self.can_do(obs, actions.FUNCTIONS.select_army.id) :
             return actions.FUNCTIONS.select_army("select")

        return actions.FUNCTIONS.no_op()
