# Get image
from matplotlib import pyplot as plt
obs = initial_state
plt.imshow(obs[f"{camera_to_use}_image"])
plt.show()

# Get depth
from matplotlib import pyplot as plt
obs = initial_state
camera_name = "robot0_eye_in_hand"
plt.imshow(obs[f"{camera_name}_depth"], cmap=plt.get_cmap('gray'))
plt.show()

# Get segmentation
from matplotlib import pyplot as plt
obs = initial_state
camera_name = "robot0_eye_in_hand"
for klass in np.unique(obs[f"{camera_name}_segmentation_element"]):from matplotlib import pyplot as plt
obs = initial_state
plt.imshow(obs[f"{camera_to_use}_image"])
plt.show()

    reshaped_img = obs[f"{camera_name}_image"].copy().reshape(256*256, 3)
    segmentation_img = obs[f"{camera_name}_segmentation_element"].flatten()
    class_0_indices = segmentation_img == klass
    reshaped_img[~class_0_indices] = np.array([255, 255, 255])
    plt.imsave(f"{klass}.png", reshaped_img.reshape(256, 256, 3))

# Plot cross hair on the image
from matplotlib import pyplot as plt
obs = initial_state
fig, ax = plt.subplots(1, 1)
ax.imshow(obs[f"{camera_to_use}_image"])
ax.plot(obj_pixel[1], 256-obj_pixel[0], '+', mew=10, ms=20)
plt.show()

# Transform from world -> pixel -> world
import robosuite.utils.camera_utils as CU
sim = env.sim
obs_dict = initial_state
obj_pos = obs_dict["object-state"][:3]
camera_name = camera_to_use
# camera frame
image = obs_dict["{}_image".format(camera_name)][::-1]

# unnormalized depth map
depth_map = obs_dict["{}_depth".format(camera_name)][::-1]
depth_map = CU.get_real_depth_map(sim=sim, depth_map=depth_map)

# get camera matrices
world_to_camera = CU.get_camera_transform_matrix(
    sim=sim,
    camera_name=camera_name,
    camera_height=env.camera_heights[0],
    camera_width=env.camera_widths[0],
)
camera_to_world = np.linalg.inv(world_to_camera)

# transform object position into camera pixel
obj_pixel = CU.project_points_from_world_to_camera(
    points=obj_pos,
    world_to_camera_transform=world_to_camera,
    camera_height=env.camera_heights[0],
    camera_width=env.camera_widths[0],
)

# transform from camera pixel back to world position
estimated_obj_pos = CU.transform_from_pixels_to_world(
    pixels=obj_pixel,
    depth_map=depth_map,
    camera_to_world_transform=camera_to_world,
)

# the most we should be off by in the z-direction is 3^0.5 times the maximum half-size of the cube
max_z_err = np.sqrt(3) * 0.022
z_err = np.abs(obj_pos[2] - estimated_obj_pos[2])
assert z_err < max_z_err

print("pixel: {}".format(obj_pixel))
print("obj pos: {}".format(obj_pos))
print("estimated obj pos: {}".format(estimated_obj_pos))
print("z err: {}".format(z_err))


# Plot crosshair on pick position
world_to_camera = CU.get_camera_transform_matrix(
    sim=env.sim,
    camera_name=camera_to_use,
    camera_height=env.camera_heights[0],
    camera_width=env.camera_widths[0],
)

obj_pixel = CU.project_points_from_world_to_camera(
    points=cloth_body_pos,
    world_to_camera_transform=world_to_camera,
    camera_height=env.camera_heights[0],
    camera_width=env.camera_widths[0],
)

# Plot cross hair on the image
from matplotlib import pyplot as plt
obs = initial_state
fig, ax = plt.subplots(1, 1)
ax.imshow(obs[f"{camera_to_use}_image"])
ax.plot(obj_pixel[1], 256-obj_pixel[0], '+', mew=10, ms=20)
plt.show()
