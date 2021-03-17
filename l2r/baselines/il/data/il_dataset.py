import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from numpy import load
from torchvision import transforms
import glob
import json
import copy
import math
import os
from .utils import sort_nicely
import ipdb as pdb

# Normalize speed to 0-1
SPEED_FACTOR = 12.0
AUGMENT_LATERAL_STEERINGS = 6

class ILDataset(Dataset):

    def __init__(self, root_dir, dataset_name, split_name, lookahead=1, preload_name=None):
        
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, dataset_name, split_name)
        self.preload_name = preload_name
        self.lookahead = 1

        if self.preload_name is not None and os.path.exists(
                os.path.join('_preloads', self.preload_name + '.npy')):
        
            print(" Loading from NPY ")
            
            self.sensor_data_names, self.measurements = np.load(
                os.path.join('_preloads', self.preload_name + '.npy'), 
                allow_pickle=True)
            
            print(self.sensor_data_names)
        
        else:
            self.sensor_data_names, self.measurements = self._preload_files(self.data_dir) 

        self.transform_op = transforms.Compose([
             transforms.Resize(256),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.measurements)
    
    def __getitem__(self, idx):
       
        #sample_path = os.path.join(self.root_dir, self.data_dir, self.sensor_data_names[idx])
        
        #input_image = Image.open(img_path)
        #image_tensor = self.transform_op(input_image)
        
        measurement = self.measurements[idx]

        try: next_measurement = self.measurements[idx+self.lookahead]
        except IndexError: next_measurement = measurement

        #mappings: http://ec2-3-90-183-136.compute-1.amazonaws.com/multimodal.html#environment-observations

        image = self.transform_op(measurement['img'])

        steering = measurement['multimodal_data'][0]

        gear = torch.FloatTensor(measurement['multimodal_data'][1])

        mode = torch.FloatTensor(measurement['multimodal_data'][2])
        
        directional_velocity = torch.FloatTensor([measurement['multimodal_data'][3],
                measurement['multimodal_data'][4],
                measurement['multimodal_data'][5]])

        directional_acceleration = torch.FloatTensor([measurement['multimodal_data'][6],
                measurement['multimodal_data'][7],
                measurement['multimodal_data'][8]])

        directional_angular_velocity = torch.FloatTensor([measurement['multimodal_data'][9],
                measurement['multimodal_data'][10],
                measurement['multimodal_data'][11]])

        yaw_pitch_roll = torch.FloatTensor([measurement['multimodal_data'][12],
                measurement['multimodal_data'][13],
                measurement['multimodal_data'][14]])

        vehicle_center_coord = torch.FloatTensor([measurement['multimodal_data'][15],
                measurement['multimodal_data'][16],
                measurement['multimodal_data'][17]]) # y, x, z

        wheel_rpm = torch.FloatTensor([measurement['multimodal_data'][18],
                measurement['multimodal_data'][19],
                measurement['multimodal_data'][20],
                measurement['multimodal_data'][21]])

        wheel_braking = torch.FloatTensor([measurement['multimodal_data'][22],
                measurement['multimodal_data'][23],
                measurement['multimodal_data'][24],
                measurement['multimodal_data'][25]])

        wheel_torque = torch.FloatTensor([measurement['multimodal_data'][26],
                measurement['multimodal_data'][27],
                measurement['multimodal_data'][28],
                measurement['multimodal_data'][29]])

        action = torch.FloatTensor(measurement['action'])
        
        # values from self.lookahead steps ahead

        next_image = self.transform_op(measurement['img'])

        next_steering = next_measurement['multimodal_data'][0]

        next_gear = torch.FloatTensor(next_measurement['multimodal_data'][1])

        next_mode = torch.FloatTensor(next_measurement['multimodal_data'][2])
        
        next_directional_velocity = torch.FloatTensor([next_measurement['multimodal_data'][3],
                next_measurement['multimodal_data'][4],
                next_measurement['multimodal_data'][5]])

        next_directional_acceleration = torch.FloatTensor([next_measurement['multimodal_data'][6],
                next_measurement['multimodal_data'][7],
                next_measurement['multimodal_data'][8]])

        next_directional_angular_velocity = torch.FloatTensor([next_measurement['multimodal_data'][9],
                next_measurement['multimodal_data'][10],
                next_measurement['multimodal_data'][11]])

        next_yaw_pitch_roll = torch.FloatTensor([next_measurement['multimodal_data'][12],
                next_measurement['multimodal_data'][13],
                next_measurement['multimodal_data'][14]])

        next_vehicle_center_coord = torch.FloatTensor([next_measurement['multimodal_data'][15],
                next_measurement['multimodal_data'][16],
                next_measurement['multimodal_data'][17]]) # y, x, z

        next_wheel_rpm = torch.FloatTensor([next_measurement['multimodal_data'][18],
                next_measurement['multimodal_data'][19],
                next_measurement['multimodal_data'][20],
                next_measurement['multimodal_data'][21]])

        next_wheel_braking = torch.FloatTensor([next_measurement['multimodal_data'][22],
                next_measurement['multimodal_data'][23],
                next_measurement['multimodal_data'][24],
                next_measurement['multimodal_data'][25]])

        next_wheel_torque = torch.FloatTensor([next_measurement['multimodal_data'][26],
                next_measurement['multimodal_data'][27],
                next_measurement['multimodal_data'][28],
                next_measurement['multimodal_data'][29]])

        next_action = torch.FloatTensor(measurement['action'])

        
        sample = {
                'image': image, 
                'steering': steering,
                'gear': gear,
                'mode': mode,
                'directional_velocity': directional_velocity,
                'directional_acceleration': directional_acceleration,
                'directional_angular_velocity': directional_angular_velocity,
                'yaw_pitch_roll': yaw_pitch_roll,
                'vehicle_center_coord': vehicle_center_coord,
                'wheel_rpm': wheel_rpm,
                'wheel_braking': wheel_braking,
                'wheel_torque': wheel_torque,
                'action': action
                }
        
        target = {
                'next_image': next_image, 
                'next_steering': next_steering,
                'next_gear': next_gear,
                'next_mode': next_mode,
                'next_directional_velocity': next_directional_velocity,
                'next_directional_acceleration': next_directional_acceleration,
                'next_directional_angular_velocity': next_directional_angular_velocity,
                'next_yaw_pitch_roll': next_yaw_pitch_roll,
                'next_vehicle_center_coord': next_vehicle_center_coord,
                'next_wheel_rpm': next_wheel_rpm,
                'next_wheel_braking': next_wheel_braking,
                'next_wheel_torque': next_wheel_torque,
                'next_action': next_action
                }


        return image, torch.FloatTensor(measurement['multimodal_data']), action


    def _preload_files(self, path):
        """
        Pre load the image folders for each episode, keep in mind that we only take
        the measurements that we think that are interesting for now.

        Args
            the path for the dataset

        Returns
            sensor data names: it is a vector with n dimensions being one for each sensor modality
            for instance, rgb only dataset will have a single vector with all the image names.
            float_data: all the wanted float data is loaded inside a vector, that is a vector
            of dictionaries.

        """

        episodes_list = sorted(glob.glob(os.path.join(path, 'episode_*')))
        sort_nicely(episodes_list)

        # Do a check if the episodes list is empty
        assert len(episodes_list) > 0, "Fatal: "+__file__+": No episodes in train set - "+ path

        sensor_data_names = []
        float_dicts = []
        number_of_hours_pre_loaded = 0

        # Now we do a check to try to find all the
        for episode in episodes_list:

            print('Episode ', episode)

            transitions_list = sorted(glob.glob(os.path.join(episode, 'transition*')))
            sort_nicely(transitions_list)

            if len(transitions_list) == 0:
                print("EMPTY EPISODE")
                continue

            # A simple count to keep track how many measurements were added this episode.
            count_added_measurements = 0

            sample = 0
            for transition in transitions_list:

                with np.load(transition) as sample:
                    float_dicts.append(sample)
                
                sensor_data_names.append(os.path.join(episode.split('/')[-1], transition))
                count_added_measurements += 1

            last_data_point_number = transitions_list[-1].split('_')[-1].split('.')[0]
            number_of_hours_pre_loaded += (float(count_added_measurements) / 3600.0)
            print(" Loaded ", number_of_hours_pre_loaded, " hours of data")

        # Make the path to save the pre loaded datasets
        if not os.path.exists('_preloads'):
            os.mkdir('_preloads')
        # If there is a name we saved the preloaded data
        if self.preload_name is not None:
            np.save(os.path.join('_preloads', self.preload_name), [sensor_data_names, float_dicts])

        return sensor_data_names, float_dicts

