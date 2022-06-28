import numpy as np
import matplotlib.pyplot as plt


def local_to_pixels_coords(local_coords, pixel_size=224):
    '''
    from (x, y) coordinates in the agent's local frame of reference ((0, 0) is the agents position)
    gives the (x, y) coords in the (pixel_size, pixel_size) frame

    in the layout image:
    - x axis origin is bottom left
    - y axis origin is top left clc

    X: right is positive, left is negative
    Y: positive is in front of the vehicle, negative is behind
    '''

    # center point in pixels
    pixel_c = (pixel_size / 2.0, pixel_size * 4.0 / 5.0)

    local_x = local_coords[0]
    local_y = local_coords[1]

    pixels_per_meter = pixel_size / 50.0   # frames we use are 50 meters wide and high

    pixel_x_diff = local_x * pixels_per_meter
    pixel_x = int(pixel_c[0] + pixel_x_diff)

    pixel_y_diff = local_y * pixels_per_meter
    pixel_y = int(pixel_c[1] - pixel_y_diff)

    return (pixel_x, pixel_y)


def compute_pixels_traj(local_traj, layout_size):
    '''
    local_traj: [nb_points, 2]
    layout_size: int representing side length of target image in pixels
    '''
    kwargs = {'pixel_size': layout_size}
    return np.apply_along_axis(local_to_pixels_coords, 1, local_traj, **kwargs)


def draw_trajectory_on_layout(layout, traj):
    '''
    layout is an image representing the scene, with the agent positioned such that:
    - 40 meters ahead
    - 10 meters behind
    - 25 meters both sides
    traj should be in local coords (n_points, 2)

    in the layout image:
    - x axis origin is bottom left
    - y axis origin is top left
    '''
    size = layout.shape[0]

    pix_traj = compute_pixels_traj(traj, size)
    all_x, all_y = zip(*pix_traj)

    plt.imshow(layout)
    plt.scatter(x=all_x, y=all_y, c='r', s=40)
    plt.show()
