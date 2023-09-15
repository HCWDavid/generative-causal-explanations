import cv2
import numpy as np
import matplotlib.pyplot as plt
#import gif

"""
surface plot

INPUTS:
    - ax : axis to draw figure on
    - x : numpy array corresponding to ROWS of Z (displayed on x-axis)
          x[0] corresponds to Z[0,:] and x[end] corresponds to Z[end,:]
    - y : numpy array corresponding to COLUMNS of Z (displayed on y-axis)
          y[0] corresponds to Z[:,0] and y[end] corresponds to Z[:,end]
    - Z : image to plot
    - clim : color limits for image; default: [min(Z), max(Z)]
"""
def plotsurface(ax, x, y, Z, clim=None):
    x = x.flatten()
    y = y.flatten()
    deltax = x[1]-x[0]
    deltay = y[1]-y[0]
    extent = (np.min(x)+deltax/2,
              np.max(x)-deltax/2,
              np.min(y)+deltay/2,
              np.max(y)-deltay/2)
    if clim == None:
        clim = [np.min(Z), np.max(Z)]
    im = ax.imshow(np.transpose(Z),
                   origin='lower',
                   extent=extent,
                   vmin=clim[0],
                   vmax=clim[1])
    return im

def explain_sample(x, save_path=None):
    # subplot for each x that has
    assert len(x.shape) == 4
    (nsamples,nrows,ncols,nchans) = x.shape
    fig, axs = plt.subplots(1, nsamples)
    for isamp,index in enumerate(range(nsamples)):
        if nchans == 1:
            axs[isamp].imshow(x[isamp,:,:,0], cmap='gray')
        else:
            axs[isamp].imshow(x[isamp,:,:,:])
        # label for each subplot:
        axs[isamp].set_title('%d' % index)
        axs[isamp].axis('off')
    if save_path is not None:
        plt.savefig(f'./{save_path}_samples_all_classes.png', dpi=500, bbox_inches='tight')
    return fig, axs

        



"""
plotExplanation - plot explanation created by GCE.explain().

Rows in output figure correspond to samples (first dimension of Xhats);
columns correspond to latent values in sweep.

:param Xhats: result from GCE.explain()
:param yhats: result from GCE.explain()
:param save_path: if provided, will export to {<save_path>_latentdimX.svg}
"""
# def plotExplanation(Xhats, yhats, save_path=None):
#     cols = [
#     [0.047, 0.482, 0.863],  # Blue
#     [1.000, 0.761, 0.039],  # Yellow
#     [0.561, 0.788, 0.227],  # Green
#     [0.898, 0.121, 0.388],  # Pink
#     [0.121, 0.388, 0.898],  # Light Blue
#     [0.745, 0.243, 0.862],  # Purple
#     [0.960, 0.498, 0.090],  # Orange
#     [0.482, 0.047, 0.863]   # Dark Blue
# ]
#     border_size = 3
#     (nsamp, z_dim, nz_sweep, nrows, ncols, nchans) = Xhats.shape
    
#     for latent_dim in range(z_dim):
#         fig, axs = plt.subplots(nsamp, nz_sweep)
#         for isamp in range(nsamp):
#             for iz in range(nz_sweep):
#                 if nchans == 1:  # Grayscale
#                     img = Xhats[isamp, latent_dim, iz, :, :, 0].squeeze()
#                     img_bordered = np.tile(np.expand_dims(np.array(cols[int(yhats[isamp, latent_dim, iz])]), (0, 1)),
#                                            (nrows + 2 * border_size, ncols + 2 * border_size, 1))
#                     img_bordered[border_size:-border_size, border_size:-border_size, :] = \
#                         np.tile(np.expand_dims(img, 2), (1, 1, 3))
#                 else:  # RGB
#                     img = Xhats[isamp, latent_dim, iz, :, :, :].squeeze()
#                     img_bordered = np.tile(np.expand_dims(np.array(cols[int(yhats[isamp, latent_dim, iz])]), (0, 1)),
#                                            (nrows + 2 * border_size, ncols + 2 * border_size, 1))
#                     img_bordered[border_size:-border_size, border_size:-border_size, :] = img
                
#                 axs[isamp, iz].imshow(img_bordered.astype(np.float32), interpolation='nearest')
#                 axs[isamp, iz].axis('off')
        
#         axs[0, round(nz_sweep / 2) - 1].set_title('Sweep latent dimension %d' % (latent_dim + 1))
        
#         if save_path is not None:
#             # plt.savefig(f'./{save_path}_latentdim{latent_dim+1}.svg', dpi=300, bbox_inches='tight')
#             plt.savefig(f'./{save_path}_latentdim{latent_dim+1}.png', dpi=500, bbox_inches='tight')

def plotExplanation(Xhats, yhats, heatmaps=None, save_path=None):
    print("Min and Max of img:", Xhats.min(), Xhats.max())
    print("Min and Max of heatmap:", heatmaps.min(), heatmaps.max())  
    heatmaps = (heatmaps - heatmaps.min()) / (heatmaps.max() - heatmaps.min())
    cols = [
        [0.047, 0.482, 0.863],  # Blue
        [1.000, 0.761, 0.039],  # Yellow
        [0.561, 0.788, 0.227],  # Green
        [0.898, 0.121, 0.388],  # Pink
        [0.121, 0.388, 0.898],  # Light Blue
        [0.745, 0.243, 0.862],  # Purple
        [0.960, 0.498, 0.090],  # Orange
        [0.482, 0.047, 0.863]   # Dark Blue
    ]
    
    border_size = 3
    (nsamp, z_dim, nz_sweep, nrows, ncols, nchans) = Xhats.shape

    for latent_dim in range(z_dim):
        fig, axs = plt.subplots(nsamp, nz_sweep)
        
        for isamp in range(nsamp):
            for iz in range(nz_sweep):
                img = Xhats[isamp, latent_dim, iz, :, :, :3].squeeze()  # assuming the last dimension is color
                
                img_bordered = np.tile(np.expand_dims(np.array(cols[int(yhats[isamp, latent_dim, iz])]), (0, 1)),
                                       (nrows + 2 * border_size, ncols + 2 * border_size, 1))
                img_bordered[border_size:-border_size, border_size:-border_size, :] = img

                axs[isamp, iz].imshow(img_bordered.astype(np.float32), interpolation='nearest')
                
                if heatmaps is not None:
                    heatmap = heatmaps[isamp, latent_dim, iz, :, :, 0]
    
                    # Normalizing the heatmap to [0, 1]
                    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
                    
                    # Make an empty array like img_bordered
                    red_heatmap = np.zeros_like(img_bordered)
                    
                    # Insert the heatmap into the red channel of the area inside the border
                    red_heatmap[border_size:-border_size, border_size:-border_size, 0] = heatmap

                    
                    # Alpha blending with the bordered image
                    alpha = 0.5  # Change this as needed
                    overlay = (img_bordered * (1 - alpha)) + (red_heatmap * alpha)
                    
                    # Clip to [0, 1] range
                    overlay = np.clip(overlay, 0, 1)
                    
                    # Display the overlay
                    axs[isamp, iz].imshow(overlay, interpolation='nearest')


                axs[isamp, iz].axis('off')
        
        axs[0, round(nz_sweep / 2) - 1].set_title(f'Sweep latent dimension {latent_dim + 1}')
        
        if save_path is not None:
            plt.savefig(f'./{save_path}_latentdim{latent_dim+1}.png', dpi=500, bbox_inches='tight')

def outline_mask(ax, mask, bounds=(0,1,0,1), color=(0,0,0,0.25)):
    # https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph
    
    x0, x1, y0, y1 = bounds
    
    # a vertical line segment is needed, when the pixels next to each other horizontally
    #   belong to diffferent groups (one is part of the mask, the other isn't)
    # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates 
    ver_seg = np.where(mask[:,1:] != mask[:,:-1])
    
    # the same is repeated for horizontal segments
    hor_seg = np.where(mask[1:,:] != mask[:-1,:])
    
    # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
    #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
    # in order to draw a discountinuous line, we add Nones in between segments
    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0]+1))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan,np.nan))
    
    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1]+1, p[0]))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan, np.nan))
    
    # now we transform the list into a numpy array of Nx2 shape
    segments = np.array(l)
    
    # now we need to know something about the image which is shown
    #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
    #   drawn with origin='lower'
    # with this information we can rescale our points
    segments[:,0] = x0 + (x1-x0) * segments[:,0] / mask.shape[1]
    segments[:,1] = y0 + (y1-y0) * segments[:,1] / mask.shape[0]
    
    # and now there isn't anything else to do than plot it
    ax.plot(segments[:,0], segments[:,1], color=color, linewidth=1)
    
    
    