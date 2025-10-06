import numpy as np
import os
from astropy.io import fits


import matplotlib.pyplot as plt

## goal of this is to sort the components of a 3 component velocity field so that the resulting velocity fields are maximally smooth

# plan - compute gradients of the velocity fields, find areas with maximum change and try to swap components there


def load_maps(run_num):
	maps_path = f'../output/fitsimages/NGC1266_maps_run{run_num}_sortmid.fits'
	fl = fits.open(maps_path)
	maps_data = fl[0].data.squeeze()

	fl.close()

	return maps_data

def calc_gradient(vel):

	vel_grady_r = vel[1:,:] - vel[:-1,:]
	vel_grady_l = vel[:-1,:] - vel[1:,:]
	vel_gradx_d = vel[:,1:] - vel[:,:-1]
	vel_gradx_u = vel[:,:-1] - vel[:,1:]

	vel_grad_total = np.zeros(vel.shape)
	vel_grad_total[:-1,:] += vel_grady_r
	vel_grad_total[1:,:] += vel_grady_l
	vel_grad_total[:,:-1] += vel_gradx_d
	vel_grad_total[:,1:] += vel_gradx_u
	#vel_grad_total = np.sqrt(vel_grad_total)

	'''vel_c2_grady_r = vel_c2[1:,:] - vel_c2[:-1,:]
				vel_c2_grady_l = vel_c2[:-1,:] - vel_c2[1:,:]
				vel_c2_gradx_d = vel_c2[:,1:] - vel_c2[:,:-1]
				vel_c2_gradx_u = vel_c2[:,:-1] - vel_c2[:,1:]
			
				vel_c2_grad_total = np.zeros(vel_c2.shape)
				vel_c2_grad_total[:-1,:] += vel_c2_grady_r**2
				vel_c2_grad_total[1:,:] += vel_c2_grady_l**2
				vel_c2_grad_total[:,:-1] += vel_c2_gradx_d**2
				vel_c2_grad_total[:,1:] += vel_c2_gradx_u**2
				vel_c2_grad_total = np.sqrt(vel_c2_grad_total)
			
				vel_c3_grady_r = vel_c3[1:,:] - vel_c3[:-1,:]
				vel_c3_grady_l = vel_c3[:-1,:] - vel_c3[1:,:]
				vel_c3_gradx_d = vel_c3[:,1:] - vel_c3[:,:-1]
				vel_c3_gradx_u = vel_c3[:,:-1] - vel_c3[:,1:]
			
				vel_c3_grad_total = np.zeros(vel_c3.shape)
				vel_c3_grad_total[:-1,:] += vel_c3_grady_r**2
				vel_c3_grad_total[1:,:] += vel_c3_grady_l**2
				vel_c3_grad_total[:,:-1] += vel_c3_gradx_d**2
				vel_c3_grad_total[:,1:] += vel_c3_gradx_u**2
				vel_c3_grad_total = np.sqrt(vel_c3_grad_total)'''

	return vel_grad_total



def plot_histogram(grad_map, name):
	grad_vals = grad_map.flatten()

	perc_95 = np.nanpercentile(grad_vals, 95)

	plt.figure(figsize=(6,6))

	plt.hist(grad_vals, bins=50)
	ymax = plt.ylim()[1]
	plt.vlines(perc_95, ymin=0, ymax=ymax, color='k')

	plt.ylim(0,ymax)

	plt.ylabel('Number', fontsize=12)
	plt.xlabel('Spaxel Gradient (km/s)', fontsize=12)
	plt.title(name)

	plt.savefig(f'../plots/{name}_gradhist.png', bbox_inches='tight')

	plt.close()


def plot_grad_map(grad_map, name):

	plt.figure(figsize=(6,6))

	plt.imshow(grad_map, vmax=400, vmin=0)
	plt.title(name)

	plt.savefig(f'../plots/{name}_gradmap.png', bbox_inches='tight')

	plt.close()

def plot_vel_map(vel_map, name):

	plt.figure(figsize=(6,6))

	plt.imshow(vel_map, vmin=-400, vmax=400, cmap='RdBu')
	plt.title(name)

	plt.savefig(f'../plots/{name}_velmap.png', bbox_inches='tight')

	plt.close()


def swap_components(maps_data):

	pix_num = maps_data[0,:,:]
	ncomp_map = maps_data[1,:,:]
	vel_c1 = maps_data[2,:,:]
	vel_c2 = maps_data[3,:,:]
	vel_c3 = maps_data[4,:,:]
	sig_c1 = maps_data[5,:,:]
	sig_c2 = maps_data[6,:,:]
	sig_c3 = maps_data[7,:,:]

	orig_c1_grad = calc_gradient(vel_c1)
	orig_c2_grad = calc_gradient(vel_c2)
	orig_c3_grad = calc_gradient(vel_c3)

	ncomp2_mask = ncomp_map == 2
	ncomp3_mask = ncomp_map == 3

	num_iterations = 1

	c1_grad = orig_c1_grad
	c2_grad = orig_c2_grad
	c3_grad = orig_c3_grad

	c1_grad_mean = [np.nanmean(c1_grad)]
	c2_grad_mean = [np.nanmean(c2_grad)]
	c3_grad_mean = [np.nanmean(c3_grad)]

	plot_histogram(c1_grad, name=f'comp1_iter0')
	plot_histogram(c2_grad, name=f'comp2_iter0')
	plot_histogram(c3_grad, name=f'comp3_iter0')

	plot_grad_map(c1_grad, name=f'comp1_iter0')
	plot_grad_map(c2_grad, name=f'comp2_iter0')
	plot_grad_map(c3_grad, name=f'comp3_iter0')

	#array that tells you if that spaxel has been attempted to be swapped before
	swap_try = np.full(pix_num.shape, False)

	for num in np.arange(1, 1+num_iterations):
		
		vel_c1_new = vel_c1.copy()
		vel_c2_new = vel_c2.copy()
		vel_c3_new = vel_c3.copy()

		c1_grad = calc_gradient(vel_c1_new)
		c2_grad = calc_gradient(vel_c2_new)
		c3_grad = calc_gradient(vel_c3_new)

		c1_thresh = 400#np.nanpercentile(c1_grad.flatten(), 90)
		c2_thresh = 400#np.nanpercentile(c2_grad.flatten(), 90)
		c3_thresh = 400#np.nanpercentile(c3_grad.flatten(), 90)

		velc1_over = c1_grad >= c1_thresh
		velc2_over = c2_grad >= c2_thresh
		velc3_over = c3_grad >= c3_thresh

		#swap c1 and c2 where both are above thresh, and ncomp = 2 

		c1_c2_swap_all = np.logical_and(velc1_over, velc2_over)
		c1_c2_swap_ncomp2 = np.logical_and(c1_c2_swap_all, ncomp2_mask)
		c1_c2_swap_ncomp2_ind = np.where(c1_c2_swap_ncomp2 == True)

		#loop through each spaxel to swap velocities and test if gradient improves
		print(len(c1_c2_swap_ncomp2_ind[0]))
		for i in range(len(c1_c2_swap_ncomp2_ind[0])):

			iycoord = c1_c2_swap_ncomp2_ind[0][i]
			ixcoord = c1_c2_swap_ncomp2_ind[1][i]

			#print(ixcoord, iycoord)

			vel_c1_new[iycoord, ixcoord] = vel_c2[iycoord, ixcoord]
			vel_c2_new[iycoord, ixcoord] = vel_c1[iycoord, ixcoord]

			c1_grad_new = calc_gradient(vel_c1_new)
			c2_grad_new = calc_gradient(vel_c2_new)

			c1_pixgrad = c1_grad[iycoord, ixcoord]
			c2_pixgrad = c2_grad[iycoord, ixcoord]
			c1_pixgrad_new = c1_grad_new[iycoord, ixcoord]
			c2_pixgrad_new = c2_grad_new[iycoord, ixcoord]

			#dont swap if sum of new pixel gradients is higher than prev
			if c1_pixgrad_new + c2_pixgrad_new > c1_pixgrad + c2_pixgrad:
				#swap back
				vel_c1_new[iycoord, ixcoord] = vel_c1[iycoord, ixcoord]
				vel_c2_new[iycoord, ixcoord] = vel_c2[iycoord, ixcoord]

		#for 3 component, try just swapping 1 and 3?

		#c1_c3_swap_all = np.logical_and(velc1_over, velc3_over)
		#c1_c3_swap_ncomp3 = np.logical_and(c1_c3_swap_all, ncomp3_mask)

		#vel_c1_new[c1_c3_swap_ncomp3] = vel_c3[c1_c3_swap_ncomp3]
		#vel_c3_new[c1_c3_swap_ncomp3] = vel_c1[c1_c3_swap_ncomp3]

		c1_grad = calc_gradient(vel_c1_new)
		c2_grad = calc_gradient(vel_c2_new)
		c3_grad = calc_gradient(vel_c3_new)

		c1_grad_mean.append(np.nanmean(c1_grad))
		c2_grad_mean.append(np.nanmean(c2_grad))
		c3_grad_mean.append(np.nanmean(c3_grad))

		#plot_histogram(c1_grad, name=f'comp1_iter{num}')
		#plot_histogram(c2_grad, name=f'comp2_iter{num}')
		#plot_histogram(c3_grad, name=f'comp3_iter{num}')

		plot_grad_map(c1_grad, name=f'comp1_iter{num}')
		plot_grad_map(c2_grad, name=f'comp2_iter{num}')
		plot_grad_map(c3_grad, name=f'comp3_iter{num}')

		plot_vel_map(vel_c1, f'c1_iter{num}')
		plot_vel_map(vel_c2, f'c2_iter{num}')
		plot_vel_map(vel_c3, f'c3_iter{num}')

		vel_c1 = vel_c1_new
		vel_c2 = vel_c2_new
		vel_c3 = vel_c3_new


	print(c1_grad_mean)
	print(c2_grad_mean)
	print(c3_grad_mean)

	#plot_vel_map(vel_c1, 'c1_swap')
	#plot_vel_map(vel_c2, 'c2_swap')
	#plot_vel_map(vel_c3, 'c3_swap')

	#top_perc






maps_data = load_maps(4)
swap_components(maps_data)


