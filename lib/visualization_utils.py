import matplotlib.pyplot as plt 
import numpy as np 
import os 


def plot_traj_projections(y_true, obstacles, y_traj=None, x0 = None, x_target = None):

    #Plot the trajectories
    fig, axs = plt.subplots(2,3)
    fig.set_size_inches(18,10)
    for i in range(3):
        if y_traj is not None: 
            axs[0,i].plot(y_traj[:,i],'r.')
        axs[0,i].plot(y_true[:,i],'b.')
        axs[0,i].set_xlabel('T')
        axs[0,i].set_ylabel('x'+str(i+1))
        
    axs[1,0].plot(y_true[:,0], y_true[:,1], 'b.')
    axs[1,0].set_xlabel('x1')
    axs[1,0].set_ylabel('x2')

    axs[1,1].plot(y_true[:,0], y_true[:,2], 'b.', label='target traj')
    axs[1,1].set_xlabel('x1')
    axs[1,1].set_ylabel('x3')
    axs[1,1].legend()

    axs[1,2].plot(y_true[:,1], y_true[:,2], 'b.', label='target traj')
    axs[1,2].set_xlabel('x2')
    axs[1,2].set_ylabel('x3')

    if y_traj is not None: 
        axs[1,0].plot(y_traj[:,0], y_traj[:,1], 'r.', label='predicted traj')
        axs[1,1].plot(y_traj[:,0], y_traj[:,2], 'r.', label='predicted traj')
        axs[1,2].plot(y_traj[:,1], y_traj[:,2], 'r.', label='predicted traj')
        axs[1,1].legend()
        

    for obs in obstacles: 
        p_obs = obs['pos']
        r_obs = obs['rad']
        circle1 = plt.Circle(p_obs[0:2], radius=r_obs)
        axs[1,0].add_patch(circle1)

        circle2 = plt.Circle(p_obs[::2], radius=r_obs)
        axs[1,1].add_patch(circle2)

        circle3 = plt.Circle(p_obs[1:], radius=r_obs)
        axs[1,2].add_patch(circle3)
        
    if x0 is not None:
        axs[1,0].plot(x0[0], x0[1], 'og')
        axs[1,1].plot(x0[0], x0[2], 'og')
        axs[1,2].plot(x0[1], x0[2], 'og')
    if x_target is not None:
        axs[1,0].plot(x_target[0], x_target[1], 'or')
        axs[1,1].plot(x_target[0], x_target[2], 'or')
        axs[1,2].plot(x_target[1], x_target[2], 'or')

    plt.show()


def plot_traj_and_obs_3d(data_sample, pred=None, save_path=None, z_layer_surface_idx=None):
    """ 
    Function to plot trajectory and corresponding sphere obstacle in 3D for one single data sample. 

    Args:
        data_sample (dict): dictionary containing start and goal position of trajectory, 
                            as well as the obstacles in the environment.
    """
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(-1,1)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    ax.zaxis.set_tick_params(labelsize=12)

    # plot sphere obstacle
    label = ''
    for i, obs in enumerate(data_sample['obstacles']): 
        pos = obs['pos']
        rad = obs['rad']
#         pos = data_sample['p_obs']
#         rad = data_sample['r_obs']
        u = np.linspace(0, 2 *np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = pos[0] + rad * np.outer(np.cos(u), np.sin(v))
        y = pos[1] + rad * np.outer(np.sin(u), np.sin(v))
        z = pos[2] + rad * np.outer(np.ones(np.size(u)), np.cos(v))
        label += 'r= {}, pos={}, '.format(np.around(rad,decimals=1), np.around(pos,decimals=1))
        if (i+1) % 2 == 0: 
            label += '\n'
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)

    # plot start and goal position 
    ax.scatter(data_sample['x0'][0], data_sample['x0'][1], data_sample['x0'][2], marker='x', s=100, c='r')
    ax.scatter(data_sample['xT'][0], data_sample['xT'][1], data_sample['xT'][2], marker='x', s=100, c='r')

    if data_sample['x_w'] is not None: 
        ax.scatter(data_sample['x_w'][0], data_sample['x_w'][1], data_sample['x_w'][2], marker='x', s=100, c='g')

    # plot trajectory 
    xs = data_sample['xs']
    ax.plot(xs[:,0],xs[:,1],xs[:,2],color='blue', label=label)

    # plot z_layer cuts: 
    if z_layer_surface_idx is not None: 
        for z_val in z_layer_surface_idx: 
            (x, y) = np.meshgrid(range(-1,1), range(-2,2))
            z  = np.zeros([*x.shape]) * z_val
            ax.plot_surface(x, y, z, alpha=0.1, color='0.8')

    # plot prediction
    if pred is not None: 
        ax.plot(pred[:,0], pred[:,1],pred[:,2], color='red')
    
    # plt.legend(prop={'size': 12})#label, loc='upper right')
    plt.tight_layout()
    plt.show()

    if save_path is not None: 
        plt.savefig(save_path, 
                dpi=300, 
                format='png', 
                bbox_inches='tight')
    return fig


def plot_env_3d(obstacles):
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    # plot sphere obstacle
    label = ''
    for obs in obstacles: 
        pos = obs['pos']
        rad = obs['rad']
#         pos = data_sample['p_obs']
#         rad = data_sample['r_obs']
        u = np.linspace(0, 2 *np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = pos[0] + rad * np.outer(np.cos(u), np.sin(v))
        y = pos[1] + rad * np.outer(np.sin(u), np.sin(v))
        z = pos[2] + rad * np.outer(np.ones(np.size(u)), np.cos(v))
        label += 'r= {}, pos={}, '.format(np.around(rad,decimals=1), np.around(pos,decimals=1))
        ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', linewidth=0, alpha=0.5)
    
    plt.legend()#label, loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_dataset_overview(trajs, legend=False, plot_obs=False, plot_all=True, num_plots=None, save_path=None): 
    fig = plt.figure()
    fig.set_size_inches(20, 10)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')

    ax.set_xlabel('x1', fontsize=16)
    ax.set_ylabel('x2', fontsize=16)
    ax.set_zlabel('x3', fontsize=16)
    ax.zaxis.set_tick_params(labelsize=14)
    ax.zaxis.set_ticks([-1.,-0.5,0.,0.5, 1.])

    ax.xaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_ticks([-1.,-0.5,0.,0.5, 1.])

    ax.yaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_ticks([-1.,-0.5,0.,0.5, 1.])

    if plot_all: 
        num_plots = len(trajs)
    colormap = plt.cm.gist_rainbow #nipy_spectral, Set1,Paired 
    colors = [colormap(i) for i in np.linspace(0, 1, num_plots)]
    labels = []

    for i,data in enumerate(trajs[100:100+num_plots]):
        # plot sphere obstacle
        label = ''
        for obs in data['obstacles']: 
            if plot_obs: 
                pos = obs['pos']
                rad = obs['rad']
                u = np.linspace(0, 2 *np.pi, 100)
                v = np.linspace(0, np.pi, 100)
                x = pos[0] + rad * np.outer(np.cos(u), np.sin(v))
                y = pos[1] + rad * np.outer(np.sin(u), np.sin(v))
                z = pos[2] + rad * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z,  rstride=4, cstride=4, color=colors[i], linewidth=0, alpha=0.5)
#             label += 'r={}, p={}; '.format(np.around(rad, decimals=1), np.around(pos,decimals=1))

        # plot trajectory 
        xs = data['xs']
        ax.plot(xs[:,0],xs[:,1],xs[:,2],color=colors[i])
        labels.append(label)

        ax.scatter(data['x0'][0], data['x0'][1], data['x0'][2], marker='x', s=100, c='r')
        ax.scatter(data['xT'][0], data['xT'][1], data['xT'][2], marker='x', s=100, c='r')
        
    if legend: 
        # lg = plt.legend(labels, loc='upper right', bbox_to_anchor=(1.6, 1.), ncol=2)    
        lg = plt.legend(labels, bbox_to_anchor=(-0.2, 1.02, 1., .102), loc='lower left', ncol=1, frameon=False, prop={'size': 12})
    
    plt.tight_layout()
    plt.show()

    if save_path is not None: 
        if legend: 
            fig.savefig(save_path, 
                dpi=300, 
                format='pdf', 
                bbox_extra_artists=(lg,), 
                bbox_inches='tight')
        else:       
            fig.savefig(save_path, 
                dpi=300, 
                format='pdf', 
                bbox_inches='tight')
    # return fig 


def plot_2d_proj_voxelgrid(voxelgrid, feature_vector, cmap='viridis'):
    z_dim = voxelgrid[2]
    fig, axes = plt.subplots(int(np.ceil(z_dim / 5)),
                             np.min((z_dim, 5)),
                             figsize=(20, 20))
    
    plt.tight_layout()
    for i, ax in enumerate(axes.flat if z_dim > 1 else [plt.gca()]):
        if i < z_dim:
            ax.imshow(feature_vector[:, :, i],
                      cmap=cmap,
                      interpolation="nearest")
            ax.set_title("Level " + str(i))
        else:
            ax.axis('off')
    color_im = ax.imshow(feature_vector[:, :, int(z_dim/2)],
                      cmap=cmap,
                      interpolation="nearest")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    fig.colorbar(color_im, cax=cbar_ax)

def show_single_layer(sdf_vol, layer_idx, save_path=None):
    fig, ax = plt.subplots(1, 1)
    plt.tight_layout()
    cmap = 'Spectral'
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.imshow(sdf_vol[:, :, layer_idx],
              cmap=cmap,
              interpolation="nearest")
    ax.set_title("SDF volume sliced along z-voxel-layer " + str(layer_idx), fontsize=18)

    color_im = ax.imshow(sdf_vol[:, :, layer_idx],
                      cmap=cmap,
                      interpolation="nearest")
    # fig.colorbar(color_im, font_size=12) 
    cbar_ax = fig.add_axes([0.9, 0.05, 0.03, 0.9])
    cbar_ax.tick_params(labelsize=14)
    fig.colorbar(color_im, cax=cbar_ax)
    plt.show()
    if save_path: 
        fig.savefig(save_path, dpi=300, 
                format='pdf', 
                bbox_inches='tight')


def compare_2vols_layers(sdf_vols, save_path=None, cmap = 'Spectral'): 
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    plt.tight_layout()
#     idx = np.round(np.linspace(0, z_dim-1, 4)).astype(int)
    
    layers = [15,35]
    locs = [(-10,-60), (-20,-5)]
    i=0
    for title, vol in sdf_vols.items():
        sdf = vol.copy()
        z_dim = vol.shape[2]
        plt.text(*locs[i], title, fontsize = 22)
        for j, layer_idx in enumerate(layers): 
            print(i,j, layer_idx)
            print(title)
            print(vol[0,0,0])

            ax = axes[i][j]
            ax.imshow(sdf[:, :, layer_idx],
                      cmap=cmap,
                      interpolation="nearest")
            ax.set_title("Layer " + str(layer_idx), fontsize=16)
        i+=1
    color_im = ax.imshow(sdf[:, :, int(z_dim/2)],
                         cmap=cmap,
                         interpolation="nearest")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
    cbar_ax.tick_params(labelsize=16)
    fig.colorbar(color_im, cax=cbar_ax)
    if save_path is not None: 
        plt.savefig(save_path, dpi=300, format='pdf', bbox_inches='tight') 
    plt.show()