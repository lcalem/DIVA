import copy
import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Function

from config.config import cfg

LAMBDA_X = 2.0
LAMBDA_Y = 1.0
LAMBDA_GRAD = 0.00001


def reconstruction_loss(predictions, targets, k, device, selection='random'):
    '''
    predictions: [bs, nsamples, nfuture, 2 (x, y)]
    targets: [bs, nfuture, 2], because there is only one GT trajectory

    k: the number of predicted trajectories (among the nsamples) in which to propagate the reconstruction loss
    selection: how those k's are selected:
        - random: duh
        - closest: take the k predicted trajectories closest to the GT trajectory
    '''
    criterion = nn.MSELoss(reduction='none')
    batch_size, h_future, n_dim = targets.shape     # future horizon (in timesteps), n_dim is the dimension of a point usually 2

    # print(f'pred shape {predictions.shape}, targets shape {targets.shape}')

    # 1. Compute MSE for each predicted trajectory
    stacked_targets = torch.tensor(np.repeat(targets, cfg.NSAMPLES, axis=0), dtype=torch.float32).to(device)
    stacked_outputs = predictions.view(-1, h_future, n_dim).to(device)
    # print(f'stacked pred shape {stacked_outputs.shape}, stacked targets shape {stacked_targets.shape}')
    # print(f'targets type {type(targets)}, stacked targets type {type(stacked_targets)}')

    mses = criterion(stacked_targets, stacked_outputs)    # [BS * NSAMPLES, h_future, 2]
    mses = mses.mean(axis=2).mean(axis=1)                 # [BS * NSAMPLES]

    # print(f'Shape of the computed MSEs : {mses.shape}, type {type(mses)}, criterion {type(criterion)}')

    if selection == "random":

        # 2. Randomly choose k elements with a Bernouilli dist
        prob = k / cfg.NSAMPLES
        binomial = torch.distributions.binomial.Binomial(probs=prob)

        # 3. Mask out non-selected elements of the loss and add everything
        masked = mses * binomial.sample(mses.shape).to(device) * (1.0 / prob)
        loss = masked.sum() / masked.count_nonzero()
        # print(f'Shape of masked losses: {masked.shape}, type: {type(masked)}, dtype: {masked.dtype} mse {mses.dtype}, {stacked_targets.dtype} {stacked_outputs.dtype}')
        # print(f'dtype loss: {loss.dtype}')

        return loss, [i % cfg.NSAMPLES for i, elt in enumerate(masked) if elt != 0]

    elif selection == "closest":
        # 2. Compute the ADE of each prediction with the GT
        ADEs = metrics.mean_distances(stacked_outputs.detach().cpu().numpy(), stacked_targets.detach().cpu().numpy())

        # 3. For each batch, find the cutoff value depending on k
        cutoff_values = 0    # [32]

        # 4. use torch.le to find the boolean values for the stacked_outputs closest to the GT
        masked_ades = ADEs * torch.le(ADEs, cutoff_values)

        # 5. loss is the mean



    else:
        raise Exception("NONONONOO")


def loss_kullback_leibler(mu, logvar):
    '''
    Reconstruction + KL divergence losses summed over all elements and batch
    see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    https://arxiv.org/abs/1312.6114
    0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    '''

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def diversity_loss(predictions, targets, diversity_kernel, device):
    '''
    DPP loss
    # predictions [batch_size, nsamples, seq_len, nfeatures]
    # target [batch_size, seq_len, nfeatures]

    kernels:
    - mse: L2 between the points
    - weighted_l2: L2 but with more weight put on the final points
    - final: L2 between the last point only
    '''
    allowed_kernels = ('mse', 'weighted_l2', 'final', 'azimuth', 'az_l2', 'az_final', 'az_weighted', 'xy', 'az2_l2')
    assert diversity_kernel in allowed_kernels, f'unknown diversity kernel {diversity_kernel}'

    criterion = torch.nn.MSELoss()
    nsamples = predictions.shape[1]

    S = torch.zeros((nsamples, nsamples)).to(device)  # similarity matrix
    for i in range(0, nsamples):
        for j in range(0, nsamples):
            if i <= j:

                if diversity_kernel == 'mse':
                    S[i, j] = criterion(predictions[:,i,:,:], predictions[:,j,:,:])
                elif diversity_kernel == 'final':
                    S[i, j] = criterion(predictions[:,i,-1,:], predictions[:,j,-1,:])
                elif diversity_kernel == 'weighted_l2':
                    S[i, j] = weighted_l2(predictions[:,i,:,:], predictions[:,j,:,:], device)
                elif diversity_kernel == 'azimuth':
                    S[i, j] = azimuth2(predictions[:,i,-1,:], predictions[:,j,-1,:])
                elif diversity_kernel == 'az_l2':
                    S[i, j] = azimuth2(predictions[:,i,-1,:], predictions[:,j,-1,:]) + criterion(predictions[:,i,:,:], predictions[:,j,:,:])
                elif diversity_kernel == 'az2_l2':
                    S[i, j] = torch.mul(2.0, azimuth2(predictions[:,i,-1,:], predictions[:,j,-1,:])) + criterion(predictions[:,i,:,:], predictions[:,j,:,:])
                elif diversity_kernel == 'az_final':
                    S[i, j] = azimuth2(predictions[:,i,-1,:], predictions[:,j,-1,:]) + criterion(predictions[:,i,-1,:], predictions[:,j,-1,:])
                elif diversity_kernel == 'az_weighted':
                    S[i, j] = azimuth2(predictions[:,i,-1,:], predictions[:,j,-1,:]) + weighted_l2(predictions[:,i,:,:], predictions[:,j,:,:], device)
                elif diversity_kernel == 'xy':
                    S[i, j] = diff_mse(predictions[:,i,:,:], predictions[:,j,:,:], criterion)

            S[j, i] = S[i, j]  # symmetric matrix

    # Kernel computation:
    S_mean = torch.mean(S)
    # print('S mean ',S_mean)

    # if diversity_kernel == 'mse':
    # TODO: see if we need another constant for other kernels
    Lambda = S_mean
    K = torch.exp(-Lambda * S)

    I = torch.eye((nsamples)).to(device)
    M = I - torch.inverse(K + I)
    dpp_loss = - torch.trace(M)
    # print('trace ', -dpp_loss)
    return dpp_loss


def test(filepath):


    with open(filepath, 'r+') as f_in:

        # patata
        b = "prout"

    # truc
    a = filepath
    a = b

    d = dict()
    d = {}

    l = list()
    l = []

    s = {'a', 'b', 'c'}
    s = set(['a', 'b', 'c'])


def weighted_l2(pred_i, pred_j, device):
    '''
    predictions [batch_size, seq_len, nfeatures]
    '''
    weights = torch.tensor([1, 2, 3, 4, 5, 6]).to(device)

    # loss = nn.MSELoss(reduce=False)
    loss = nn.MSELoss(reduction='none')
    output = loss(pred_i, pred_j)

    end_sum = torch.sum(output, dim=-1)
    weighted_sum = torch.mul(end_sum, weights)    # weights are broadcasted

    bs, seq_len, dim = output.shape
    n = torch.mul(torch.mul(torch.sum(weights), bs), dim)   # * bs * dim (to get all the weights)

    total = torch.div(torch.sum(weighted_sum), n)
    return total


def diff_mse(pred_i, pred_j, mse):
    '''
    differentiated MSE between X and Y
    hyperparam: balancing between x and y
    '''
    pred_i_x = pred_i[:,:,0]
    pred_j_x = pred_j[:,:,0]

    pred_i_y = pred_i[:,:,1]
    pred_j_y = pred_j[:,:,1]

    mse_x = mse(pred_i_x, pred_j_x)
    mse_y = mse(pred_i_y, pred_j_y)

    return torch.add(torch.mul(LAMBDA_X, mse_x), torch.mul(LAMBDA_Y, mse_y))


def azimuth(pred_i_final, pred_j_final):
    '''
    compares azimuth of final points
    angle = scalar product / product of magnitudes

    example on one angle between p1 (pred_i), (0,0), and p2 (pred_j):
    (x1−x3)(x2−x3)+(y1−y3)(y2−y3) / sqrt((x1−x3)^2+(y1−y3)^2) * sqrt((x2−x3)^2+(y2−y3)^2)
    conveniently, (x3,y3) is (0,0)
    '''

    scalar_products = torch.add(torch.mul(pred_i_final[:,0], pred_j_final[:,0]),torch.mul(pred_i_final[:,1], pred_j_final[:,1]))
    magnitudes = torch.mul(torch.sqrt(torch.add(torch.square(pred_i_final[:,0]),torch.square(pred_i_final[:,1]))),torch.sqrt(torch.add(torch.square(pred_j_final[:,0]),torch.square(pred_j_final[:,1]))))
    # print(magnitudes)

    cos_angles = torch.div(scalar_products, magnitudes)
    cos_angles = torch.clamp(cos_angles, min=-1.0, max=1.0)  # small numerical float imprecisions can make the values fo off-bound, screwing up acos
    angles = torch.acos(cos_angles)
    print('cos_angles and angles')
    print(cos_angles)
    print(angles)
    r = torch.sum(angles)
    print(r)
    return r


def azimuth2(pred_i_final, pred_j_final):
    cossim = 1.0 - torch.nn.functional.cosine_similarity(pred_i_final, pred_j_final, dim=1, eps=1e-8)
    s = torch.sum(cossim)
    # print(s)
    return s


# --------------------------- LAYOUT LOSS

class LayoutDist(Function):
    '''
    TODO: voir s'il faut pas spécifier le grad par rapport aux inputs dont on se fout
    '''

    @staticmethod
    def forward(ctx, pixel_coords, map_idx, diff_map, dx_map, dy_map, device): # coords.shape: [BS * NT * n_future, 2]
        # output [BS * NT * n_future, 1] <- 1: distance
        batch_size = pixel_coords.shape[0]
        # print("LayoutDist batch size %s" % batch_size)

        # print("In LayoutDist, pixel_coords shape")
        # print(pixel_coords.shape)

        # print("In LayoutDist, map_idx shape")
        # print(map_idx.shape)
        # aller chercher la valeur de la distance pour chaque (x, y) prédit

        pixel_dist = torch.zeros((batch_size, 1)).to(device)

        for i_batch in range(batch_size): # loop over all (x, y) in the superbatch (BS * NT * n_future)
            # print(f'for i_batch {i_batch}, diff_map i is {map_idx[i_batch]}')
            dist = diff_map[map_idx[i_batch], pixel_coords[i_batch][1].to(int), pixel_coords[i_batch][0].to(int)].to(device)
            # print(f'for i_batch {i_batch}, coords {pixel_coords[i_batch]}, dist {dist}')

            pixel_dist[i_batch, :] = dist

        # save precomputed map gradients
        ctx.save_for_backward(diff_map, dx_map, dy_map, pixel_coords, map_idx)

        # print("In LayoutDist, pixel_dist shape")
        # print(pixel_dist.shape)
        return pixel_dist

    @staticmethod
    def backward(ctx, grad_output):
        # sortie delta x et delta y de dx_map, dy_map
        # return [BS * NT * n_future, 2] <- dx, dy

        GRAD_MULT = copy.copy(LAMBDA_GRAD)
        if cfg.DSF.LAMBDA is not None:
            GRAD_MULT = LAMBDA_GRAD * (1.0 - cfg.DSF.LAMBDA)

        device = grad_output.device
        diff_map, dx_map, dy_map, pixel_coords, map_idx = ctx.saved_tensors

        # print(f"Hello this is backward shapes {grad_output.shape}")

        # print(f"Hello this is backward ctx shapes {dx_map.shape} {dy_map.shape}, {map_idx.shape}")

        batch_size, _ = grad_output.shape
        Hessian = torch.zeros((batch_size, 2)).to(device)    # return [BS * NT * n_future, 2] <- dx, dy
        for i_batch in range(batch_size):
            # TODO: retrouver (x, y) à partir de la loss via save_for_grad
            x, y = pixel_coords[i_batch]

            diff = diff_map[map_idx[i_batch], pixel_coords[i_batch][1].to(int), pixel_coords[i_batch][0].to(int)]

            dx = torch.mul(GRAD_MULT, dx_map[map_idx[i_batch], pixel_coords[i_batch][1].to(int), pixel_coords[i_batch][0].to(int)])
            dy = torch.mul(GRAD_MULT, dy_map[map_idx[i_batch], pixel_coords[i_batch][1].to(int), pixel_coords[i_batch][0].to(int)])

            # dx = dx_map[map_idx[i_batch], pixel_coords[i_batch][1].to(int), pixel_coords[i_batch][0].to(int)]
            # dy = dy_map[map_idx[i_batch], pixel_coords[i_batch][1].to(int), pixel_coords[i_batch][0].to(int)]
            Hessian[i_batch,:] = torch.FloatTensor([dy, dx]).to(device)

            # print(f'for i {i_batch}, coords are {x, y} and grad is {[dx, dy]} and diff is {diff}')

        return Hessian, None, None, None, None, None


def layout_loss(predictions, diff_mask, dx_mask, dy_mask, device):
    '''
    Shapes:
    diff_mask, dx_mask, dy_mask    [BS, 224, 224]
    predictions                    [BS * 12 * 6, 2]

    output                         [BS * 12 * 6, 1]
    '''
    pixel_size = torch.Tensor([diff_mask.shape[-1]]).to(float).to(device)
    bs, npred, npoints, dim = predictions.shape
    # predictions = torch.clamp(predictions, min=-25.0, max=25.0)
    assert dim == 2

    # center point in pixels
    pixel_c = torch.Tensor([torch.div(pixel_size, 2), torch.div(torch.mul(pixel_size, 4.0), 5.0)]).to(device)
    pixels_per_meter = torch.Tensor([torch.div(pixel_size, 50.0)]).to(device)   # frames we use are 50 meters wide and high
    xy_coefs = torch.Tensor([1.0, -1.0]).to(device)  # to get the final pixel vector we need to add -y and +x to the center point

    # accumulated distance of pixels outside DA
    loss = torch.tensor(0.0).to(device)

    # convert predictions from local to pixel frame of reference to match the mask
    pixcoords = torch.zeros_like(predictions).to(device)   # [BS, NPRED, NPOINTS, 2]
    map_idx = torch.zeros((bs, npred, npoints), dtype=torch.long).to(device)
    for i_batch in range(bs):
        for i_pred, p_traj in enumerate(predictions[i_batch]):

            for i_point, local_coords in enumerate(p_traj):

                # difference to center point in pixels (locals are in meter)
                diff_coords = torch.mul(local_coords, pixels_per_meter)
                diff_coords = torch.mul(diff_coords, xy_coefs)

                # absolute pixel coords (from pixel difference centered on agent)
                pixel_coords = torch.add(pixel_c, diff_coords)
                pixel_coords = torch.clamp(pixel_coords, min=0, max=diff_mask.shape[-1])

                # stack pixel coords
                pixcoords[i_batch, i_pred, i_point] = pixel_coords.to(device)
                map_idx[i_batch, i_pred, i_point] = i_batch

    # discrete_coords = torch.nn.functional.grid_sample(float_coords, diff_mask)

    # get the distance value from the diffmap with the discrete coords
    # print(f'pixel coords shape: {pixcoords.shape}')
    distance_da = LayoutDist.apply
    dist = distance_da(pixcoords.view(-1, 2), map_idx.view(-1, 1), diff_mask, dx_mask, dy_mask, device)
    # print(f'dist shape {dist.shape}')
    # print(f'dist shape {torch.mean(dist).shape}')

    loss += torch.mean(dist)

    # print(f'layout loss for batch: {loss}')
    return loss


def old_layout_loss(predictions, da_mask, device):
    '''
    for each predicted pixel out of the drivable area, add 1 to the layout loss
    loss is accumulated over all predictions

    Shapes:
    da_mask      [BS, 1, 224, 224]
    predictions  [BS, 12, 6, 2]

    output       [BS, 1]
    '''

    # print(f'mask shape {da_mask.shape}')
    # print(f'pred shape 1 {predictions.shape}')
    # print(f'unique {torch.unique(da_mask)}')

    pixel_size = torch.Tensor([da_mask.shape[-1]]).to(float).to(device)
    bs, npred, npoints, dim = predictions.shape
    # predictions = torch.clamp(predictions, min=-25.0, max=25.0)
    assert dim == 2

    # center point in pixels
    pixel_c = torch.Tensor([torch.div(pixel_size, 2), torch.div(torch.mul(pixel_size, 4.0), 5.0)]).to(device)
    pixels_per_meter = torch.Tensor([torch.div(pixel_size, 50.0)]).to(device)   # frames we use are 50 meters wide and high
    xy_coefs = torch.Tensor([1.0, -1.0]).to(device)  # to get the final pixel vector we need to add -y and +x to the center point

    # accumulated number of pixels outside DA
    acc_loss = torch.tensor(0.0).to(device)

    # convert predictions from local to pixel frame of reference to match the mask
    for i_batch in range(bs):
        for p_traj in predictions[i_batch]:

            for local_coords in p_traj:

                # difference to center point in pixels (locals are in meter)
                diff_coords = torch.mul(local_coords, pixels_per_meter)
                diff_coords = torch.mul(diff_coords, xy_coefs)

                # absolute pixel coords (from pixel difference centered on agent)
                pixel_coords = torch.add(pixel_c, diff_coords)
                pixel_coords = torch.clamp(pixel_coords, min=0, max=da_mask.shape[-1])

                mask_value = da_mask[i_batch, 0, pixel_coords[1].to(int), pixel_coords[0].to(int)]
                acc_loss += mask_value

    # print(f'layout loss for batch: {acc_loss}')
    return acc_loss
