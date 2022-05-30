import numpy as np
import habitat
import time
from habitat.core.dataset import Dataset, Episode, EpisodeIterator
import pdb
import torch
import habitat_sim
import klampt
import pandas as pd
from klampt.math import so3,se3
# from klampt_transforms import Transform_tool
import pickle
from matplotlib import pyplot as plt
import cv2
import gc
import rospy
from std_msgs.msg import String
import quaternion
from habitat_sim.utils import common as utils
import queue
from threading import Lock,Thread
import faulthandler
# from agents.utils.semantic_prediction import SemanticPredMaskRCNN
faulthandler.enable()

class Semantic_env:
    def __init__(self,path_to_hm3d = "/home/motion/habitat-challenge/habitat-challenge-data/data/scene_datasets/hm3d"):
        self.path_to_hm3d = path_to_hm3d
        self.q = queue.Queue()
        self.started = False
        self.names_to_classes = {'chair':0,'couch':5,'plant':2,'bed':1,'toilet':3,'tv':4,'table':6,'kitchen table':6,'oven':7,'sink':8,'refrigerator':9,'book':10,'clock':11,'vase':12,'flower vase':12,'cup':13,'bottle':14,'soap bottle':14}

        self.create_scene_at()
        self.imgs_dir = '/home/motion/data/semantic_evaluation_sem_exp/{}/{}/{}.{}'
    def create_scene_at(self,scene_id = "vLpv2VX547B"):
        if self.started:
            del self.sim
            del self.sim_cfg
            del self.scene
            gc.collect()
        self.scene_id = scene_id
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.scene_id = scene_id #f"{path_to_hm3d}/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
        backend_cfg.scene_dataset_config_file = f"{self.path_to_hm3d}/hm3d_annotated_basis_actual.scene_dataset_config.json"
        # print('\n\n\n\n\n\n',dir(backend_cfg),'\n\n\n\n\n\n\n')
        # pdb.set_trace()
        # sem_cfg = habitat_sim.CameraSensorSpec()
        # sem_cfg.uuid = "semantic"
        # sem_cfg.sensor_type = habitat_sim.SensorType.SEMANTIC
        settings = {'width':640,'height':480,'sensor_height':0.88}
        sensor_specs = []
        color_sensor_spec = habitat_sim.CameraSensorSpec()
        color_sensor_spec.uuid = "color_sensor"
        color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_sensor_spec.resolution = [settings["height"], settings["width"]]
        color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        color_sensor_spec.hfov = 79
        color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(color_sensor_spec)

        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [settings["height"], settings["width"]]
        depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        depth_sensor_spec.hfov = 79
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_spec)

        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
        semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
        semantic_sensor_spec.hfov = 79
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(semantic_sensor_spec)


        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = sensor_specs

        sim_cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
        self.sim_cfg = sim_cfg
        self.sim = habitat_sim.Simulator(sim_cfg)

        self.scene = self.sim.semantic_scene
        self.index_dict,self.category_to_index = self.create_index_dict(self.scene)
        self.colors = self.create_random_colors()

        

        print("# objects: {}".format(len(self.sim.semantic_scene.objects)))
        self.started = True

    def create_index_dict(self,scene):
        index_dict={}
        category_to_index = {}
        # pdb.set_trace()
        dct = self.names_to_classes
        test_dict = {}
        for obj in scene.objects:
            category_to_index.update({obj.category.name():obj.category.index()})
            index_dict.update({obj.semantic_id:dct.get(obj.category.name().lower(),255)})
        for category in scene.categories:
            test_dict.update({category.name():category.index()})
            # print('semantic id = {} | category index = {} | category name = {}'.format(obj.semantic_id,obj.category.index(),obj.category.name()))
        # print('category to index: {} \n | \n categories : {}'.format(category_to_index,test_dict))

        # print index_dict
        return index_dict,category_to_index

    def listen(self):
        self.node = rospy.init_node('gt_listener',anonymous= True)
        self.sub = rospy.Subscriber('/env_state_and_others',String,callback = self.save_gt_image,queue_size = 1000)
        # self.saving_thread = Thread(target = self.serve_all)
        self.serve_all()
        # self.saving_thread.start()
        # rospy.spin()
    
    def save_gt_image(self,data):
        msg = data.data
        self.q.put(msg)

        # cv2.imshow('gt_image',semantic_colored)
        # cv2.waitKey(1)

    def serve_all(self):
        while True:
            msg = self.q.get()
            pos,rot,scene_name,imcounter,thold = msg.split('|')
            pos = np.array(eval(pos)).astype(float)
            rot = quaternion.quaternion(*eval(rot))
            # rot = utils.quat_to_magnum(rot)
            if(self.scene_id != scene_name):
                self.reconfigure_sim(scene_name)
            state = self.sim.agents[0].get_state()
            state.position = pos
            state.rotation = rot
            self.sim.agents[0].set_state(state)
            observations = self.sim.get_sensor_observations()
            semantic = self.map_instance_to_class(observations['semantic_sensor'])
            semantic = semantic.astype(np.uint8)
            semantic_colored = self.colors[semantic]
            cv2.imwrite(self.imgs_dir.format(thold,'gt',imcounter,'png'),semantic)
            cv2.imshow('gt_image',semantic_colored)
            cv2.waitKey(1)



    def map_class(self,idx):
        return self.index_dict.get(idx,255)


    def map_instance_to_class(self,idxs):
        return np.vectorize(self.map_class)(idxs)

    def create_random_colors(self,CLASSES = 256*[0]):
        np.random.seed(42)
        COLORS = np.random.randint(0, 256, size=(len(CLASSES) - 1, 3),dtype="uint8")
        COLORS = np.vstack([[0, 0, 0], COLORS]).astype("uint8")
        return COLORS

    def reconfigure_sim(self,new_scene_id):
        # pdb.set_trace()
        self.sim_cfg.sim_cfg.scene_id = new_scene_id
        self.scene_id = new_scene_id
        # cfg = self.sim.config.backend_c
        # cfg.sim_cfg.scene_id = new_scene_id
        # self.sim.reconfigure(cfg)
        # self.sim.__(self.sim_cfg)
        self.sim._Simulator__set_from_config(self.sim_cfg)
        # self.sim.curr_scene_name = new_scene_id
        # pdb.set_trace()
        self.sim.reset()
        # self.sim.step_world()
        # pdb.set_trace()
        self.scene = self.sim.semantic_scene
        self.index_dict,self.category_to_index = self.create_index_dict(self.scene)


    def drive_and_show(self):
        observations = self.sim.reset()
        cv2.imshow('color image', cv2.cvtColor(observations['color_sensor'][:,:,:3], cv2.COLOR_RGB2BGR))

        while(True):
            keyboard_input = cv2.waitKey(1)
            if keyboard_input in (27, ord('q'), ord('Q')):
                break
            elif keyboard_input == ord('w'):
                action = 'move_forward'
            elif keyboard_input == ord('a'):
                action = 'turn_left'
            elif keyboard_input == ord('d'):
                action = 'turn_right'


            if(keyboard_input in [ord('a'),ord('w'),ord('d')]):
                # try:

                observations = self.sim.step(action)
                semantic = self.map_instance_to_class(observations['semantic_sensor'])
                # print(np.unique([observations['semantic_sensor']]))

                semantic = semantic.astype(np.uint8)
                semantic_colored = self.colors[semantic]

                # theta = observations['compass'][0]
                # x,y = observations['gps']
                # rotation_matrix = tf.transformations.rotation_matrix(theta,(0,0,1))
                states = self.sim.get_agent(0).state
                position = states.sensor_states['color_sensor'].position 
                orientation = states.sensor_states['color_sensor'].rotation.components
                orientation = orientation/np.linalg.norm(orientation)

                # compass.append(theta)
                # gps.append(observations['gps'])
                # x,y = observations['gps']
                # x = x + 0.88*np.sin(theta)
                # y = y + 0.88*np.cos(theta)
                # print(observations['gps'])
                cv2.imshow('color image', cv2.cvtColor(observations['color_sensor'][:,:,:3], cv2.COLOR_RGB2BGR))
                cv2.imshow('depth image',(observations['depth_sensor']*255/5).astype(np.uint8))
                cv2.imshow('semantic image',semantic_colored)
                # except Exception as e:
            
                #     break
        
    def get_observation_at(self,state):
        self.sim.agents[0].set_state(state)
        # observations = self.sim.step('STOP')
        
        observations = self.sim.get_sensor_observations()
        semantic = self.map_instance_to_class(observations['semantic_sensor'])
        semantic = semantic.astype(np.uint8)
        # semantic_colored = self.colors[semantic]
        return semantic


#  mapped_semantic = vectorize(semantic)

if __name__ == '__main__':
    import faulthandler

    faulthandler.enable()
    from matplotlib import pyplot as plt
    import numpy as np
    import time
    import torch
    from habitat.core.dataset import Dataset, Episode, EpisodeIterator
    import cv2

    import habitat_sim
    from klampt.math import so3,se3
    from habitat.datasets import make_dataset


    
    # Load embodied AI task (RearrangePick) and a pre-specified virtual robot
    # config=habitat.get_config("/home/motion/habitat-challenge/configs/challenge_objectnav2022_semantic.local.rgbd.yaml")
    # config.defrost()
    # config['SIMULATOR']['SCENE'] = "vLpv2VX547B"
    # config['SIMULATOR']['DEPTH_SENSOR']['NORMALIZE_DEPTH'] = False
    # config['SIMULATOR']['SCENE_DATASET'] = '/home/motion/habitat-challenge/habitat-challenge-data/data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json'

    # config.freeze()
    # dataset = make_dataset(
    #             id_dataset=config.DATASET.TYPE, config=config.DATASET
            # )
    # print('n\\n\n\nMADE THE FUCKING DATASET\n\n\n')
    # args = pickle.load(open('./seg_args.pkl','rb'))
    # model = SemanticPredMaskRCNN(args)

    # print('starting this shit')
    sim = Semantic_env()
    # pdb.set_trace()
    sim.listen()
    # # pdb.set_trace()
    # sim.drive_and_show()
    # for scn in ["vLpv2VX547B","qk9eeNeR4vw","oEPjPNSPmzL","gmuS7Wgsbrx","ixTj1aTMup2","Wo6kuutE9i7","6imZUJGRUq4","3XYAD64HpDr","Jfyvj3xn2aJ"]:


    #     sim.reconfigure_sim(scn)
    #     sim.drive_and_show()
    # obs = sim.sim.step('turn_left')
    # plt.imshow(obs['color_sensor'])
    # plt.show()
    # print(sim.sim.curr_scene_name)
    # sim.create_scene_at('Wo6kuutE9i7')
    # obs = sim.sim.step('turn_left')
    # plt.imshow(obs['color_sensor'])
    # plt.show()
    # print(sim.sim.curr_scene_name)
    # pdb.set_trace()
