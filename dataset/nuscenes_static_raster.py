import numpy as np

from pyquaternion import Quaternion

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

from nuscenes.prediction.input_representation.interface import StaticLayerRepresentation
from nuscenes.prediction.input_representation.static_layers import load_all_maps, get_crops, correct_yaw

from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction.helper import angle_of_rotation


def hex_to_rgb(hexa):
    '''
    input:  #B4FBB8
    output: (180, 251, 184)
    '''
    h = hexa.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def bmask_to_rgb(bmask, color):
    '''
    input (H, W) binary mask (0: white, 1: black)
    
    1. adds last dimension to 2D binary mask
    2. the 1 from the binary mask are set to the given color
    3. the 0 from the binary mask are set to white in the RGB mask
    returns the constructed RGB mask (H, W, C)
    '''
    init_shape = bmask.shape
    assert len(init_shape) == 2, f'expected 2D binary mask, got {len(init_shape)} dimensions'
    stacked = np.stack((bmask,) * 3, axis=-1)
    assert stacked.shape == (init_shape[0], init_shape[1], 3)
    
    # adds color for 1s
    colored_stacked = stacked * color
    
    # transfer from binary masks where 0 is white to RGB mask where white is (255, 255, 255)
    mask = np.where((colored_stacked[:, :, 0] == 0) & (colored_stacked[:, :, 1] == 0) & (colored_stacked[:, :, 2] == 0))
    colored_stacked[mask] = (255, 255, 255)

    return colored_stacked.astype('uint8')


def average_color(all_imgs):
    '''
    Average of each image pasted onto one image
    '''
    average_color = np.zeros_like(all_imgs[0], np.float)

    for im in all_imgs:
        assert im.shape == average_color.shape
        average_color += im / len(all_imgs)

    return average_color.astype(np.uint8)   # important to cast to uint8 because other wise floats are clipped to [0, 1]


class StaticLayerRasterizerCustom(StaticLayerRepresentation):
    """
    Creates a representation of the static map layers where
    the map layers are given a color and rasterized onto a
    three channel image.
    """

    def __init__(self,
                 helper,
                 layer_names=None,
                 colors=None,
                 resolution=0.1,  # meters / pixel
                 meters_ahead=40,
                 meters_behind=10,
                 meters_left=25,
                 meters_right=25):

        self.helper = helper
        self.maps = load_all_maps(helper)

        if not layer_names:
            layer_names = ['drivable_area', 'ped_crossing', 'walkway']
        self.layer_names = layer_names

        self.resolution = resolution
        self.meters_ahead = meters_ahead
        self.meters_behind = meters_behind
        self.meters_left = meters_left
        self.meters_right = meters_right

    def get_angle_in_degrees(self, rotation):
        yaw = quaternion_yaw(Quaternion(rotation))
        yaw_corrected = correct_yaw(yaw)
        return angle_of_rotation(yaw_corrected) * 180 / np.pi

    def make_representation(self, instance_token, sample_token):
        '''
        Rasterized representation of static map layers.

        :param instance_token: Token for instance.
        :param sample_token: Token for sample.
        :return: Three channel image.
        '''
        sample_annotation = self.helper.get_sample_annotation(instance_token, sample_token)
        map_name = self.helper.get_map_name_from_sample_token(sample_token)
        nusc_map = self.maps[map_name]

        x, y = sample_annotation['translation'][:2]

        image_side_length = 2 * max(self.meters_ahead, self.meters_behind, self.meters_left, self.meters_right)
        image_side_length_pixels = int(image_side_length / self.resolution)

        mask_patchbox = (x, y, image_side_length, image_side_length)

        angle_in_degrees = self.get_angle_in_degrees(sample_annotation['rotation'])

        canvas_size = (image_side_length_pixels, image_side_length_pixels)

        masks = nusc_map.get_map_mask(mask_patchbox, angle_in_degrees, self.layer_names, canvas_size=canvas_size)

        # get color for each mask (assuming masks are in the same order than layers)
        color_map = nusc_map.explorer.color_map
        assert len(masks) == len(self.layer_names)
        colored_masks = list()
        for i in range(len(self.layer_names)):
            flipped_mask = masks[i, ::-1, :]
            # print(f'flipped mask for layer {layers[i]}')
            # plt.imshow(flipped_mask, interpolation='nearest')
            # plt.show()

            color = hex_to_rgb(color_map[self.layer_names[i]])
            colored_masks.append(bmask_to_rgb(flipped_mask, color))

        # print('colored mask for layer -1')
        # plt.imshow(colored_masks[-1])
        # plt.show()

        # blit all colored masks on one final image
        image = average_color(colored_masks)

        # Crop the scene
        row_crop, col_crop = get_crops(self.meters_ahead, self.meters_behind, self.meters_left, self.meters_right, self.resolution, int(image_side_length / self.resolution))

        out_img = image[row_crop, col_crop, :]
        # print('final image for static')
        # plt.imshow(out_img, interpolation='nearest')
        # plt.show()

        da_mask = masks[0, ::-1, :]                # flip it (like for creating the original image)
        da_mask = da_mask[row_crop, col_crop]      # crop it

        return out_img, da_mask


class InputRepresentationCustom:
    """
    Specifies how to represent the input for a prediction model.
    Need to provide a StaticLayerRepresentation - how the map is represented,
    an AgentRepresentation - how agents in the scene are represented,
    and a Combinator, how the StaticLayerRepresentation and AgentRepresentation should be combined.

    Takes a StaticLayerRasterizerCustom to be able to get the drivable area mask (da_mask)
    """

    def __init__(self, 
                 static_layer,
                 agent,
                 combinator):

        self.static_layer_rasterizer = static_layer
        self.agent_rasterizer = agent
        self.combinator = combinator

    def make_input_representation(self, instance_token: str, sample_token: str) -> np.ndarray:

        static_layers, da_mask = self.static_layer_rasterizer.make_representation(instance_token, sample_token)
        agents = self.agent_rasterizer.make_representation(instance_token, sample_token)

        return (self.combinator.combine([static_layers, agents]), da_mask)
