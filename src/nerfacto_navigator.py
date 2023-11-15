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
from nerfdiff.utils.nerf_utils import load_nerfacto_model
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


class NerfactoNavigator():
    def __init__(self, config_path='./cfg/llff_global.yaml'):
        self.config = ReadConfig(path = config_path)
        self.initialize()
        self.get_initial_distribution()

        # if self.log_results:
        #     # If using a provided start we already have ground truth, so don't log redundant gt.
        #     if not self.use_logged_start:
        #         with open(self.log_directory + "/" + "gt_" + self.model_name + "_" + str(self.obs_img_num) + "_" + "poses.npy", 'wb') as f:
        #             np.save(f, self.gt_pose)

        #     # Add initial pose estimate before first update step is run.
        #     if self.use_weighted_avg:
        #         position_est = self.filter.compute_weighted_position_average()
        #     else:
        #         position_est = self.filter.compute_simple_position_average()
        #     rot_est = self.filter.compute_simple_rotation_average()
        #     pose_est = gtsam.Pose3(rot_est, position_est).matrix()
        #     self.all_pose_est.append(pose_est)

    def initialize(self):
        self.num_updates = 0
        self.timestamp = get_timestamp()
        self.particles_timeline = []

        self.use_convergence_protection = self.config.get_param('use_convergence_protection')
        self.use_weighted_avg = self.config.get_param('use_weighted_avg')
        self.number_convergence_particles = self.config.get_param('number_convergence_particles')
        self.convergence_noise = self.config.get_param('convergence_noise')
        self.use_received_image = self.config.get_param('use_received_image')
        self.sampling_strategy = self.config.get_param('sampling_strategy')
        self.ckpt_dir = self.config.get_param('ckpt_dir')
        self.data_dir = self.config.get_param('data_dir')
        self.batch_size = self.config.get_param('batch_size')
        self.num_particles = self.config.get_param('num_particles')
        self.min_bounds = self.config.get_param('min_bounds')
        self.max_bounds = self.config.get_param('max_bounds')
        self.min_number_particles = self.config.get_param('min_number_particles')
        self.use_particle_reduction = self.config.get_param('use_particle_reduction')
        self.use_refining = self.config.get_param('use_refining')
        self.plot_particles  = self.config.get_param('visualize_particles')
        self.plot_particles_interval  = self.config.get_param('visualize_particles_interval')
        self.log_directory = self.config.get_param('log_directory')
        
        self.run_predicts = self.config.get_param('run_predicts')



        self.alpha_refine = self.config.get_param('alpha_refine')
        self.alpha_super_refine = self.config.get_param('alpha_super_refine')

        self.nerf_model = load_nerfacto_model(Path(self.ckpt_dir))
        self.device = self.nerf_model.device
        self.dataset = SceneDataset(self.data_dir, device=self.device, filename='poses_test_15.txt')
        self.metadata = self.dataset.metadata
        subsample_method = 'sift'
        subsample_n_points = 100
        subsample_dilation = 5
        image_sampler = ImageSubsampler(subsample_method, subsample_n_points, subsample_dilation)
        
        
        eval_index = 0
        self.target = self.dataset[eval_index]['image']
        self.target_image = self.target.permute(1, 2, 0)
        self.target_pose = self.dataset[eval_index]['qctc']
        (mask,), (n_points_in_mask,) = image_sampler.get_mask(self.target[None], True)
        self.mask = torch.tensor(mask)

        self.visualize_output_directory = Path(self.log_directory) / self.timestamp
        self.visualize_output_directory.mkdir(parents=True, exist_ok=True)
        pass
    def get_initial_distribution(self):
        self.initial_particles = self.get_initial_particles()
        self.filter = ParticleFilter(self.initial_particles)

    def get_initial_particles(self):
        initial_particles_noise = np.random.uniform(
            np.array([self.min_bounds['px'], self.min_bounds['py'], self.min_bounds['pz'], self.min_bounds['rz'], self.min_bounds['ry'], self.min_bounds['rx']]),
            np.array([self.max_bounds['px'], self.max_bounds['py'], self.max_bounds['pz'], self.max_bounds['rz'], self.max_bounds['ry'], self.max_bounds['rx']]),
            size = (self.num_particles, 6))

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
            # pdb.set_trace()
            # print(initial_particles)
        return {'position':initial_positions, 'rotation':np.array(rots)}

    
    def rgb_callback(self, msg):
        self.img_msg = msg
        
    def rgb_run(self):
        start_time = time.time()

        # make copies to prevent mutations
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
        
        quats = np.array([ rotmat2qvec(x.matrix()) for x in self.filter.particles['rotation']])

        particles = np.concatenate((quats, self.filter.particles['position']), axis=-1)
        self.particles_timeline.append(particles)
        particles = torch.tensor(particles)
        losses = get_photometric_error_nerfacto(
            particles=particles,
            target_images=self.target,
            nerf=self.nerf_model,
            meta=self.metadata,
            device=self.device,
            masks=self.mask,
            no_of_rays=self.batch_size*self.num_particles)
        
        # pdb.set_trace()

        for index, particle in enumerate(particles_position_before_update):
                self.filter.weights[index] = 1/losses[index]

        self.filter.update()
        self.num_updates += 1
        print("UPDATE STEP NUMBER", self.num_updates, "RAN")
        print("number particles:", self.num_particles)

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

        if self.plot_particles and self.num_updates % self.plot_particles_interval == 0:
            self.visualize(particles, pose_est)
            
        # TODO add ability to render several frames
        # if self.view_debug_image_iteration != 0 and (self.num_updates == self.view_debug_image_iteration):
        #     self.nerf.visualize_nerf_image(self.nerf_pose)

        # if not self.use_received_image:
        #     if self.use_weighted_avg:
        #         print("average position of all particles: ", self.filter.compute_weighted_position_average())
        #         print("position error: ", np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_weighted_position_average()))
        #     else:
        #         print("average position of all particles: ", self.filter.compute_simple_position_average())
        #         print("position error: ", np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_simple_position_average()))

        # if self.use_weighted_avg:
        #     position_est = self.filter.compute_weighted_position_average()
        # else:
        #     position_est = self.filter.compute_simple_position_average()
        # rot_est = self.filter.compute_simple_rotation_average()
        # pose_est = gtsam.Pose3(rot_est, position_est).matrix()

        # if self.log_results:
        #     self.all_pose_est.append(pose_est)
        
        # if not self.run_inerf_compare:
        #     img_timestamp = msg.header.stamp
        #     self.publish_pose_est(pose_est, img_timestamp)
        # else:
        #     self.publish_pose_est(pose_est)
    
        update_time = time.time() - start_time
        print("Time taken:", update_time)

        if not self.run_predicts:
            self.filter.predict_no_motion(self.px_noise, self.py_noise, self.pz_noise, self.rot_x_noise, self.rot_y_noise, self.rot_z_noise) #  used if you want to localize a static image
        
        # return is just for logging
        return pose_est
    
    def check_if_position_error_good(self, return_error = False):
        """
        check if position error is less than 5cm, or return the error if return_error is True
        """
        acceptable_error = 0.05
        if self.use_weighted_avg:
            error = np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_weighted_position_average())
            if return_error:
                return error
            return error < acceptable_error
        else:
            error = np.linalg.norm(self.gt_pose[0:3,3] - self.filter.compute_simple_position_average())
            if return_error:
                return error
            return error < acceptable_error

    def check_if_rotation_error_good(self, return_error = False):
        """
        check if rotation error is less than 5 degrees, or return the error if return_error is True
        """
        acceptable_error = 5.0
        average_rot_t = (self.filter.compute_simple_rotation_average()).transpose()
        # check rot in bounds by getting angle using https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices

        r_ab = average_rot_t @ (self.gt_pose[0:3,0:3])
        rot_error = np.rad2deg(np.arccos((np.trace(r_ab) - 1) / 2))
        print("rotation error: ", rot_error)
        if return_error:
            return rot_error
        return abs(rot_error) < acceptable_error
    
    def set_noise(self, scale):
        self.px_noise = self.config.get_param('px_noise') / scale
        self.py_noise = self.config.get_param('py_noise') / scale
        self.pz_noise = self.config.get_param('pz_noise') / scale
        self.rot_x_noise = self.config.get_param('rot_x_noise') / scale
        self.rot_y_noise = self.config.get_param('rot_y_noise') / scale
        self.rot_z_noise = self.config.get_param('rot_z_noise') / scale
        # pass


    def check_refine_gate(self):
    
        # get standard deviation of particle position
        sd_xyz = np.std(self.filter.particles['position'], axis=0)
        norm_std = np.linalg.norm(sd_xyz)
        refining_used = False
        print("sd_xyz:", sd_xyz)
        print("norm sd_xyz:", np.linalg.norm(sd_xyz))

        if norm_std < self.alpha_super_refine:
            print("SUPER REFINE MODE ON")
            # reduce original noise by a factor of 4
            self.set_noise(scale = 4.0)
            refining_used = True
        elif norm_std < self.alpha_refine:
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

    def visualize(self, particles, pose_est):
        fig = plt.figure(figsize=(24,12))
        grid = fig.add_gridspec(nrows=3, ncols=2)

        #! Plot Final Particles
        final_particles = particles.clone().detach().cpu().numpy()
        gt = self.target_pose.clone().detach().cpu().numpy()

        ax = fig.add_subplot(grid[:2,0], projection='3d')
        ax.view_init(elev=-70, azim=90, roll=0)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)

        final_particles = utils.normalize_qt(final_particles) # Normalize q
        cam_orn = (-1,1,-1)

        # Plot Camera Poses on this time step
        plot_cam_poses(final_particles, ax, cam_orientation=cam_orn, scale=0.02, alpha=0.3)
        plot_cam_poses(gt[None], ax, cam_orientation=cam_orn, scale=0.02, alpha=1, color='green')
        
        fig_save_path = self.visualize_output_directory / f'{self.num_updates}.png'
        plt.savefig(fig_save_path)
        plt.close()
        pass



if __name__ == '__main__':
    navigator = NerfactoNavigator('./cfg/nerfacto.yaml')
    start_time = time.time()
    for i in range(50):
        pose_estimate = navigator.rgb_run()
        print(pose_estimate)
    print(navigator.target_pose)
    print(f'Time taken = {time.time() - start_time}')

 