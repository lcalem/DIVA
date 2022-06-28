import numpy as np


def all_local_to_pixels(local_trajs, map_h, pixel_c, pixels_per_meter):
    '''
    converts all x and all y from a bunch of trajectories into pixel coords
    '''
    all_x = list()
    all_y = list()

    for i in range(local_trajs.shape[0]):
        xs, ys = local_to_pixels(local_trajs[i], map_h, pixel_c, pixels_per_meter)

        # if any(xs < 0) or any(xs > map_h) or any(ys < 0) or any(ys > map_h):
        #     print("LOCAL TRAJS")
        #     print(local_trajs)
        #     print("xs")
        #     print(xs)
        #     print("ys")
        #     print(ys)
        #     continue

        all_x.extend(xs)
        all_y.extend(ys)

    return all_x, all_y


def local_to_pixels(local_traj, map_h, pixel_c, pixels_per_meter):
    '''
    takes a trajectory (npoints, 2) in local frame of reference
    returns all_x and all_y positions in pixels
    '''
    x_local = local_traj[:, 0].flatten()
    y_local = local_traj[:, 1].flatten()

    diff_xs = x_local * pixels_per_meter
    diff_ys = y_local * pixels_per_meter * -1

    # absolute pixel coords (from pixel difference centered on agent)
    pixel_xs = diff_xs + pixel_c[0]
    pixel_ys = diff_ys + pixel_c[1]

    xs = np.array(list(map(int, pixel_xs)))
    ys = np.array(list(map(int, pixel_ys)))

    xs = np.clip(xs, 0, map_h)
    ys = np.clip(ys, 0, map_h)
    return xs, ys


def dao(pred_trajs, da_map):
    '''
    original DAO from Park et al
    https://github.com/kami93/CMU-DATF/blob/5b3f88e7d083817b3ef2c43e68579975b54581e0/Proposed/utils.py

    pred_trajs = (nb_predictions, nb_timesteps, 2)
    we don't have the agents dimension

    pred_trajs are in local frame of reference
    da_map is in pixels and has 1.0 outside of the DA

    TODO: remove trajectories that go out of the map (compare with OOM management in Park DAO)
    '''
    SCALING_FACTOR = 10000   # from the original paper, scaling factor to get a better view of the value

    map_h, map_w = da_map.shape
    assert map_h == map_w
    pixel_c = (map_h / 2, map_h * 4.0 / 5.0)
    pixel_per_meter = map_h / 50.0

    da_mask = da_map > 0

    total_da_pixels = (da_map == 0).sum().item()
    all_positions_x, all_positions_y = all_local_to_pixels(pred_trajs, map_h, pixel_c, pixel_per_meter)

    # only consider unique pixels
    all_positions = set(zip(all_positions_x, all_positions_y))
    unique_x, unique_y = zip(*all_positions)

    nb_pixels_in_da = (da_map[unique_y, unique_x] < 1.0).sum().item()
    dao = (nb_pixels_in_da * SCALING_FACTOR) / total_da_pixels

    # print(f'DAO Computation pixels in DA: {nb_pixels_in_da}, total pixels: {total_da_pixels}')

    return dao


def dac(pred_trajs, da_map):
    '''
    original DAC from ???
    pred_trajs = (nb_predictions, nb_timesteps, 2)
    da_map is in pixels and has 1.0 outside of the DA

    TODO: remove trajectories that go out of the map (compare with OOM management in Park DAO)
    '''
    map_h, map_w = da_map.shape
    assert map_h == map_w
    pixel_c = (map_h / 2, map_h * 4.0 / 5.0)
    pixel_per_meter = map_h / 50.0

    da_mask = da_map > 0

    k = pred_trajs.shape[0]
    m = 0.0

    # for each trajectory check if it goes outside of the drivable area
    for i in range(k):
        positions_x, positions_y = local_to_pixels(pred_trajs[i], map_h, pixel_c, pixel_per_meter)

        nb_pixels_outside_da = (da_map[positions_y, positions_x] > 0).sum()

        if nb_pixels_outside_da > 0:
            m += 1.0

    dac = (k - m) / k
    return dac
