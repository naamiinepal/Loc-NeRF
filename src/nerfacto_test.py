

from particle_filter import ParticleFilter
from navigator_base import ReadConfig
import numpy as np
import pdb
from nerfdiff.utils.base import normalize_qt
import time
import torch

from nerfdiff.dataset.scene_dataset import SceneDataset
from nerfdiff.utils.nerf_utils import load_nerfacto_model
from nerfdiff.utils.nerf_utils import get_photometric_error_nerfacto
from pathlib import Path
from nerfdiff.nerf.inerf import ImageSubsampler
import gtsam
import tqdm
from nerfdiff.utils.read_write_model import rotmat2qvec


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

def get_initial_particles(config):
    min_bounds = config.get_param('min_bounds')
    max_bounds = config.get_param('max_bounds')
    num_particles = config.get_param('num_particles')

    initial_particles_noise = np.random.uniform(
        np.array([min_bounds['px'], min_bounds['py'], min_bounds['pz'], min_bounds['rz'], min_bounds['ry'], min_bounds['rx']]),
        np.array([max_bounds['px'], max_bounds['py'], max_bounds['pz'], max_bounds['rz'], max_bounds['ry'], max_bounds['rx']]),
        size = (num_particles, 6))

    initial_positions = np.zeros((num_particles, 3))
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
        # print(initial_particles)
    return {'position':initial_positions, 'rotation':np.array(rots)}
    # return {'position': initial_particles_distribution[:,:3], 'rotation': initial_particles_distribution[:,3:]}

if __name__ == '__main__':
    config = ReadConfig(path ='./cfg/nerfacto.yaml')
    initial_particles = get_initial_particles(config)
    filter = ParticleFilter(initial_particles)
    use_convergence_protection = config.get_param('use_convergence_protection')
    use_weighted_avg = config.get_param('use_weighted_avg')
    number_convergence_particles = config.get_param('number_convergence_particles')
    convergence_noise = config.get_param('convergence_noise')
    use_received_image = config.get_param('use_received_image')
    sampling_strategy = config.get_param('sampling_strategy')
    ckpt_dir = config.get_param('ckpt_dir')
    data_dir = config.get_param('data_dir')
    batch_size = config.get_param('batch_size')
    num_particles = config.get_param('num_particles')


    nerf_model = load_nerfacto_model(Path(ckpt_dir))
    device = nerf_model.device
    dataset = SceneDataset(data_dir, device=device, filename='poses_eval.txt')
    metadata = dataset.metadata
    subsample_method = 'sift'
    subsample_n_points = 100
    subsample_dilation = 5
    image_sampler = ImageSubsampler(subsample_method, subsample_n_points, subsample_dilation)
    
    
    eval_index = 0
    target = dataset[eval_index]['image']
    target_image = target.permute(1, 2, 0)
    target_pose = dataset[eval_index]['qctc']
    (mask,), (n_points_in_mask,) = image_sampler.get_mask(target[None], True)
    mask = torch.tensor(mask)
    print(f'Target Pose: {target_pose}')
    # pdb.set_trace()
    for iteration in range(20):
        start_time = time.time()

        # make copies to prevent mutations
        particles_position_before_update = np.copy(filter.particles['position'])
        particles_rotation_before_update = np.copy(filter.particles['rotation'])

        if use_convergence_protection:
            for i in range(number_convergence_particles):
                t_x = np.random.uniform(low=-convergence_noise, high=convergence_noise)
                t_y = np.random.uniform(low=-convergence_noise, high=convergence_noise)
                t_z = np.random.uniform(low=-convergence_noise, high=convergence_noise)
                # TODO this is not thread safe. have two lines because we need to both update
                # particles to check the loss and the actual locations of the particles
                filter.particles["position"][i] = filter.particles["position"][i] + np.array([t_x, t_y, t_z])
                particles_position_before_update[i] = particles_position_before_update[i] + np.array([t_x, t_y, t_z])
        
        # yprs = np.array([x.ypr() for x in filter.particles['rotation']])
        # quats = np.array([euler_to_quaternion(ypr[0], ypr[1], ypr[2]) for ypr in yprs])
        quats = np.array([ rotmat2qvec(x.matrix()) for x in filter.particles['rotation']])
        # pdb.set_trace()

        particles = np.concatenate((quats, filter.particles['position']), axis=-1)
        particles = torch.tensor(particles)
        losses = get_photometric_error_nerfacto(
            particles=particles,
            target_images=target,
            nerf=nerf_model,
            meta=metadata,
            device=device,
            masks=mask,
            no_of_rays=batch_size*num_particles)
        
        pdb.set_trace()

        for index, particle in enumerate(particles_position_before_update):
                filter.weights[index] = 1/losses[index]

        if use_weighted_avg:
                avg_pose = filter.compute_weighted_position_average()
        else:
            avg_pose = filter.compute_simple_position_average()

        avg_rot = filter.compute_simple_rotation_average()
        avg_rot_ypr = avg_rot.ypr()
        quat = euler_to_quaternion(avg_rot_ypr[0], avg_rot_ypr[1], avg_rot_ypr[2])
        # pdb.set_trace()
    
        print(f'{iteration}. {list(quat) + list(avg_pose)} {time.time()-start_time}')
    print(target_pose)
    # print(avg_rot)
    # pdb.set_trace()
    
    # weights = [float(x) for x in lv]


    # if sampling_strategy == 'random':
    #     rand_inds = np.random.choice(nerf.coords.shape[0], size=nerf.batch_size, replace=False)
    #     batch = nerf.coords[rand_inds]

