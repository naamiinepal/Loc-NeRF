from thirdparty.locnerf.src.particle_filter import ParticleFilter
from thirdparty.locnerf.src.navigator_base import ReadConfig
import numpy as np
import gtsam
import time
import cv2
import torch
import glob
import matplotlib.pyplot as plt
from nerfdiff.dataset.scene_dataset import SceneDataset
from nerfdiff.utils.nerf_utils import get_photometric_error_nerfacto
from pathlib import Path
from nerfdiff.nerf.inerf import ImageSubsampler
from nerfdiff.utils.read_write_model import rotmat2qvec
import pdb
import matplotlib.pyplot as plt
import nerfdiff.utils.base as utils
from nerfdiff.utils.plot import plot_cam_poses
from nerfdiff.utils.base import get_timestamp

def euler_to_quaternion(yaw, pitch, roll):
    """
    Convert yaw, pitch, roll angles to quaternion.
    Args:
        yaw (float): Yaw angle in radians.
        pitch (float): Pitch angle in radians.
        roll (float): Roll angle in radians.
    Returns:
        np.array: Quaternion [w, x, y, z].
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return np.array([w, x, y, z])

def quaternion_to_euler(quaternion):
    """
    Convert quaternion to yaw, pitch, roll angles.
    Args:
        quaternion (np.array): Quaternion [w, x, y, z].
    Returns:
        Tuple[float]: Yaw, pitch, roll angles in radians.
    """
    w, x, y, z = quaternion

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x**2 + y**2)
    roll_x = np.arctan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)
    pitch_y = np.arcsin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y**2 + z**2)
    yaw_z = np.arctan2(t3, t4)

    return np.array([yaw_z, pitch_y, roll_x])




class NerfactoNavigator():
    def __init__(
            self,
            eval_index,
            nerf_model,
            dataset,
            num_particles,
            noise_scale=1,
            sample_around_gt=False,
            gt_qctc= None,
            initial_qctcs = None,
            num_pixels_per_particle = 200,
            use_convergence_protection = True,
            convergence_noise = 0.05,
            use_refining = True,
            noise=[0.01, 0.01, 0.01, 0.001, 0.004, 0.001],
            use_mask=False,
            debug = False
        ):
        self.config = ReadConfig(path = './thirdparty/locnerf/cfg/nerfacto.yaml')
        self.dataset = dataset
        self.eval_index = eval_index
        self.nerf_model = nerf_model
        self.num_particles = num_particles
        self.noise_scale = noise_scale
        self.sample_around_gt = sample_around_gt
        if gt_qctc:
            self.ground_truth = np.array(list(gt_qctc[-3:]) + list(quaternion_to_euler(gt_qctc[:4])))
        # pdb.set_trace()
        self.initial_qctcs = initial_qctcs
        self.batch_size = num_pixels_per_particle
        self.use_convergence_protection = use_convergence_protection
        self.convergence_noise = convergence_noise
        self.use_refining = use_refining
        self.noise=noise
        self.use_mask = use_mask
        self.debug = debug
        self.device = nerf_model.device

        self.initialize()
        self.get_initial_distribution()


    def initialize(self):
        self.num_updates = 0
        self.timestamp = get_timestamp()

        self.use_weighted_avg = self.config.get_param('use_weighted_avg')
        self.number_convergence_particles = self.config.get_param('number_convergence_particles')
        self.use_received_image = self.config.get_param('use_received_image')
        self.sampling_strategy = self.config.get_param('sampling_strategy')
        # self.num_particles = self.config.get_param('num_particles')
        self.min_bounds = self.config.get_param('min_bounds')
        self.max_bounds = self.config.get_param('max_bounds')
        self.min_number_particles = self.config.get_param('min_number_particles')
        self.use_particle_reduction = self.config.get_param('use_particle_reduction')
        self.plot_particles  = self.config.get_param('visualize_particles')
        self.plot_particles_interval  = self.config.get_param('visualize_particles_interval')
        self.log_directory = self.config.get_param('log_directory')
        
        self.run_predicts = self.config.get_param('run_predicts')

        self.alpha_refine = self.config.get_param('alpha_refine')
        self.alpha_super_refine = self.config.get_param('alpha_super_refine')

        self.device = self.nerf_model.device
        self.metadata = self.dataset.metadata

        self.target = self.dataset[self.eval_index]['image'].to(self.device)
        self.target_image = self.target.permute(1, 2, 0)
        self.target_pose = self.dataset[self.eval_index]['qctc']
        self.mask = None

        if self.use_mask:
            subsample_method = 'sift'
            subsample_n_points = 100
            subsample_dilation = 5
            image_sampler = ImageSubsampler(subsample_method, subsample_n_points, subsample_dilation)
            (mask,), (n_points_in_mask,) = image_sampler.get_mask(self.target[None], True)
            self.mask = torch.tensor(mask)

        self.visualize_output_directory = Path(self.log_directory) / self.timestamp / f'{self.eval_index}'
        self.visualize_output_directory.mkdir(parents=True, exist_ok=True)
        pass
    def get_initial_distribution(self):
        self.initial_particles = self.get_initial_particles()
        self.filter = ParticleFilter(self.initial_particles)

    def save_distribution_plot(self, fig_save_path, title=''):
        fig = plt.figure(figsize=(6,6))
        grid = fig.add_gridspec(nrows=1, ncols=1)
        ax = fig.add_subplot(grid[:2,0], projection='3d')
        ax.view_init(elev=-70, azim=90, roll=0)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.set_title(title)

        final_particles = utils.normalize_qt(self.get_particle_qctcs()) # Normalize q
        cam_orn = (-1,1,-1)
        ground_truth = self.target_pose.clone().detach().cpu().numpy()

        # Plot Camera Poses on this time step
        plot_cam_poses(final_particles, ax, cam_orientation=cam_orn, scale=0.02, alpha=0.3)
        plot_cam_poses(ground_truth[None], ax, cam_orientation=cam_orn, scale=0.02, alpha=1, color='green')
        
        plt.savefig(fig_save_path)
        plt.close()

    def get_initial_particles(self):
        if self.initial_qctcs is not None:
            initial_positions = self.initial_qctcs[:, -3:]
            rots = []
            # qctcs = self.initial_qctcs.clone().detach().cpu().numpy()
            for qctc in self.initial_qctcs:
                [yaw, pitch, roll] = quaternion_to_euler(qctc[:4])
                rots.append(gtsam.Rot3.Ypr(yaw, pitch, roll))

        else:
            initial_particles_noise = np.random.uniform(
                np.array([self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz'], self.min_bounds['rz'], self.min_bounds['ry'], self.min_bounds['rx']]),
                np.array([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz'], self.max_bounds['rz'], self.max_bounds['ry'], self.max_bounds['rx']]),
                size = (self.num_particles, 6)) * self.noise_scale
            if self.sample_around_gt:
                initial_particles_noise += self.ground_truth
                # samples = np.random.normal(0, self.particle_sampling_sigma, self.num_particles*6)
                # initial_particles_noise = samples.reshape(-1, 6) 
                # # Scale the noise so that translation noise is in range -1 to 1 and rotation noise in -pi - pi
                # initial_particles_noise[:3]*=3.141492654
                # initial_particles_noise+= self.ground_truth
            initial_positions = np.zeros((self.num_particles, 3))
            rots = []
            for index, particle in enumerate(initial_particles_noise):
                x = particle[0]
                y = particle[1]
                z = particle[2]
                phi = particle[3]
                theta = particle[4]
                psi = particle[5]
                
                # set positions
                initial_positions[index,:] = initial_particles_noise[index, :3]
                # set orientations
                rots.append(gtsam.Rot3.Ypr(phi, theta, psi))
        
        return {'position':initial_positions, 'rotation':np.array(rots)}

    def get_particle_qctcs(self):
        quats = np.array([ rotmat2qvec(x.matrix()) for x in self.filter.particles['rotation']])
        qctcs = np.concatenate((quats, self.filter.particles['position']), axis=-1)
        return qctcs

    def rgb_callback(self, msg):
        self.img_msg = msg

    def get_photometric_error(self):
        particles = torch.tensor(self.get_particle_qctcs(), device=self.device)
        # pdb.set_trace()

        losses = get_photometric_error_nerfacto(
            particles=particles,
            target_images=self.target,
            nerf=self.nerf_model,
            meta=self.metadata,
            device=self.device,
            masks=self.mask,
            no_of_rays=self.batch_size*self.num_particles)
        return losses
        
    def rgb_run(self):
        start_time = time.time()

        # make copies to prevent mutations
        # pdb.set_trace()
        particles_position_before_update = np.copy(self.filter.particles['position'])
        particles_rotation_before_update = np.copy(self.filter.particles['rotation'])

        if self.use_convergence_protection:
            for i in range(self.number_convergence_particles):
                t_x = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
                t_y = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
                t_z = np.random.uniform(low=-self.convergence_noise, high=self.convergence_noise)
                # TODO this is not thread safe. have two lines because we need to both update
                # particles to check the loss and the actual locations of the particles
                self.filter.particles["position"][i] = self.filter.particles["position"][i] + np.array([t_x, t_y, t_z])
                particles_position_before_update[i] = particles_position_before_update[i] + np.array([t_x, t_y, t_z])

        
        losses = self.get_photometric_error()

        for index, particle in enumerate(particles_position_before_update):
                self.filter.weights[index] = 1/losses[index]

        self.filter.update()
        self.num_updates += 1

        if self.use_refining: # TODO make it where you can reduce number of particles without using refining
            self.check_refine_gate()

        if self.use_weighted_avg:
                avg_pose = self.filter.compute_weighted_position_average()
        else:
            avg_pose = self.filter.compute_simple_position_average()

        avg_rot = self.filter.compute_simple_rotation_average()
        avg_rot_ypr = avg_rot.ypr()
        quat = euler_to_quaternion(avg_rot_ypr[0], avg_rot_ypr[1], avg_rot_ypr[2])
        pose_est = np.array(list(quat)+list(avg_pose))
    
        update_time = time.time() - start_time
        if self.debug:
            print("Time taken:", update_time)

        if not self.run_predicts:
            self.filter.predict_no_motion(*self.noise) #  used if you want to localize a static image
        
        # return is just for logging
        return pose_est, losses
    
    def set_noise(self, scale):
        self.px_noise = self.noise[0] / scale
        self.py_noise = self.noise[1] / scale
        self.pz_noise = self.noise[2] / scale
        self.rot_x_noise = self.noise[3] / scale
        self.rot_y_noise = self.noise[4] / scale
        self.rot_z_noise = self.noise[5] / scale
        # pass


    def check_refine_gate(self):
    
        # get standard deviation of particle position
        sd_xyz = np.std(self.filter.particles['position'], axis=0)
        norm_std = np.linalg.norm(sd_xyz)
        refining_used = False
        if self.debug:
            print("sd_xyz:", sd_xyz)
            print("norm sd_xyz:", np.linalg.norm(sd_xyz))

        if norm_std < self.alpha_super_refine:
            if self.debug:
                print("SUPER REFINE MODE ON")
            # reduce original noise by a factor of 4
            self.set_noise(scale = 4.0)
            refining_used = True
        elif norm_std < self.alpha_refine:
            if self.debug:
                print("REFINE MODE ON")
            # reduce original noise by a factor of 2
            self.set_noise(scale = 2.0)
            refining_used = True
        else:
            # reset noise to original value
            self.set_noise(scale = 1.0)
        
        if refining_used and self.use_particle_reduction:
            self.filter.reduce_num_particles(self.min_number_particles)
            self.num_particles = self.min_number_particles

    def publish_pose_est(self, pose_est_gtsam, img_timestamp = None):
        pass



if __name__ == '__main__':
    navigator = NerfactoNavigator(eval_index = 0, model_config_path='/usr/nerfacto/room15_a/config.yml', dataset_path='/usr/dataset/room15_a')
    start_time = time.time()
    for i in range(50):
        pose_estimate = navigator.rgb_run()
        print(pose_estimate)
    print(navigator.target_pose)
    print(f'Time taken = {time.time() - start_time}')

 